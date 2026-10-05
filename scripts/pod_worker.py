"""Script que roda DENTRO do Pod RunPod (SSH, sem Docker/serverless).

Mesma lógica de modelo que `runpod/*/handler.py`, mas sem o envelope de
job do RunPod Serverless e sem R2: cada subcomando lê/escreve arquivo
local, e processa um lote de itens numa só chamada (o modelo carrega uma
vez por invocação do processo, não por item) — evita recarregar
Demucs/WhisperX/Qwen/MOSS-TTS a cada segmento.

`scripts/dub.py`, rodando na máquina do desenvolvedor, copia os arquivos
de entrada pro Pod (scp/rsync), chama este script via SSH, e copia o
resultado de volta. Ver `runpod/README.md` sobre por que a versão
serverless (R2 + job JSON) ainda não foi validada end-to-end e por que a
validação de hipótese do pipeline usa o Pod direto por enquanto.

Uso:
    python pod_worker.py separate_stems --input audio.wav --out-dir stems/
    python pod_worker.py transcribe --input vocals.wav --output transcript.json
    python pod_worker.py translate --input jobs.json --output candidates.json
    python pod_worker.py synthesize --input jobs.json --out-dir synth/
    python pod_worker.py evaluate --input jobs.json --output eval.json
    python pod_worker.py match_emotion --input jobs.json --output emotion.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def cmd_separate_stems(args: argparse.Namespace) -> None:
    import demucs.api

    separator = demucs.api.Separator(model="htdemucs", device="cuda")
    _original, separated = separator.separate_audio_file(args.input)
    vocals = separated["vocals"]
    background = separated["drums"] + separated["bass"] + separated["other"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    demucs.api.save_audio(vocals, str(out_dir / "vocals.wav"), samplerate=separator.samplerate)
    demucs.api.save_audio(
        background, str(out_dir / "background.wav"), samplerate=separator.samplerate
    )


def _patch_torch_compat_shims() -> None:
    """pyannote.audio==3.1.1 (pinado transitivamente pelo whisperx 3.2.0) usa a
    API antiga de backend do torchaudio, removida em versões novas. Chamar
    antes de qualquer `import whisperx` (mesmo shim de `runpod/transcribe/_compat.py`,
    duplicado aqui pra este script não depender de um pacote local chamado
    `runpod`, que colidiria com o SDK `runpod` instalado via pip).

    Achado real 2026-09-25 (Pod Blackwell, torch/torchaudio forçado pra
    2.9.1+cu128 — só essa versão tem kernel pra sm_120): torchaudio 2.9.1 não
    só depreciou `get_audio_backend`/`set_audio_backend`/`list_audio_backends`
    (torchaudio 2.2.2 antigo só avisava) — removeu o MÓDULO
    `torchaudio.backend` inteiro E a classe `torchaudio.AudioMetaData` (não
    só moveu de lugar, sumiu de verdade). `pyannote.audio` usa
    `torchaudio.AudioMetaData` como type hint em `core/io.py` (avaliado na
    hora de importar, não só documentação) e `from torchaudio.backend.common
    import AudioMetaData` em `tasks/segmentation/mixins.py`. Sem essas duas
    coisas existirem, o import quebra antes mesmo de tentar rodar nada.
    Recria um `AudioMetaData` mínimo (mesmos campos da classe real, já
    estável há várias versões) e um módulo `torchaudio.backend.common` falso
    apontando pra ele — suficiente pro import e pro uso que pyannote faz.
    """
    import dataclasses
    import sys
    import types

    import torchaudio

    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: "soundfile"
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *args, **kwargs: None
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]

    if not hasattr(torchaudio, "AudioMetaData"):

        @dataclasses.dataclass
        class AudioMetaData:
            sample_rate: int
            num_frames: int
            num_channels: int
            bits_per_sample: int = 16
            encoding: str = "PCM_S"

        torchaudio.AudioMetaData = AudioMetaData

    if not hasattr(torchaudio, "backend"):
        backend_module = types.ModuleType("torchaudio.backend")
        common_module = types.ModuleType("torchaudio.backend.common")
        common_module.AudioMetaData = torchaudio.AudioMetaData
        backend_module.common = common_module
        torchaudio.backend = backend_module
        sys.modules["torchaudio.backend"] = backend_module
        sys.modules["torchaudio.backend.common"] = common_module

    # Achado real 2026-09-25, mesmo Pod: torch 2.6+ mudou o default de
    # `torch.load` pra `weights_only=True` (segurança). O checkpoint do
    # pyannote (carregado via `lightning_fabric.utilities.cloud_io`) tem
    # objetos customizados (`torch.torch_version.TorchVersion`,
    # `pyannote.audio.core.task.Specifications`, e provavelmente mais —
    # foram aparecendo um de cada vez ao permitir na lista manualmente) que
    # não estão na lista segura por padrão. Em vez de alistar cada classe
    # (lightning_fabric passa `weights_only=True` explicitamente, então só
    # dar allowlist não bastava — via `add_safe_globals` sozinho não
    # resolveu), força `weights_only=False` pra qualquer `torch.load`
    # depois deste ponto — aceitável aqui porque são checkpoints oficiais
    # do HuggingFace (pyannote, whisperx), não arquivo de origem
    # desconhecida.
    import functools

    import torch

    if not getattr(torch.load, "_novoaudio_patched", False):
        _original_torch_load = torch.load

        @functools.wraps(_original_torch_load)
        def _torch_load_trusted_default(*args, **kwargs):
            kwargs["weights_only"] = False
            return _original_torch_load(*args, **kwargs)

        _torch_load_trusted_default._novoaudio_patched = True
        torch.load = _torch_load_trusted_default


def cmd_transcribe(args: argparse.Namespace) -> None:
    _patch_torch_compat_shims()
    import whisperx
    from whisperx.diarize import DiarizationPipeline

    device = "cuda"
    language = "en"

    asr_model = whisperx.load_model("large-v3", device, compute_type="float16", language=language)
    align_model, align_metadata = whisperx.load_align_model(
        language_code=language, device=device
    )
    diarize_model = DiarizationPipeline(use_auth_token=os.environ["HF_TOKEN"], device=device)

    audio = whisperx.load_audio(args.input)
    result = asr_model.transcribe(audio, batch_size=16)
    result = whisperx.align(
        result["segments"], align_model, align_metadata, audio, device, return_char_alignments=False
    )
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    Path(args.output).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


def cmd_translate(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from packages.pipeline.translation import (
        N_CANDIDATES,
        build_prompt,
        parse_candidates,
        rank_candidates,
    )

    model_id = "Qwen/Qwen2.5-7B-Instruct"
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Geração em lote de um causal LM precisa de padding à esquerda (alinha
    # o FIM de todos os prompts na mesma posição, senão cada linha começaria
    # a gerar num índice diferente) e de um pad_token (Qwen não define um
    # por padrão).
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device
    )

    # Achado real 2026-09-18 (pedido explícito do usuário: reduzir tempo de
    # ponta a ponta sem perder qualidade): `cmd_translate` fazia 1 chamada
    # de `generate()` POR SEGMENTO, serial — para um vídeo com 90+
    # segmentos, isso paga o overhead de forward-pass do zero 90 vezes,
    # mesmo achado de T0.14b (gerar em lote custa quase o mesmo tempo de
    # GPU que gerar 1) nunca tinha sido aplicado ENTRE segmentos, só entre
    # candidatos do mesmo segmento (`cmd_synthesize`). Agora processa em
    # lotes de até `TRANSLATE_BATCH_SIZE` segmentos por chamada. Lote
    # limitado (não "todos de uma vez") de propósito — sem medição real de
    # uso de memória num vídeo de 90+ segmentos com prompts longos, um lote
    # sem limite arrisca CUDA OOM; 16 é uma margem de segurança, não medida
    # no limite.
    TRANSLATE_BATCH_SIZE = 16

    # Qwen às vezes não devolve JSON válido (achado real, 2x em bateladas de
    # 90+ segmentos) — sem retry, 1 job ruim derrubava a tradução inteira do
    # vídeo, mesmo com todos os outros já traduzidos certo. Reamostra (a
    # geração é `do_sample=True`, então tentar de novo é uma chance real de
    # sucesso, não repetir o mesmo erro) até MAX_TRANSLATE_ATTEMPTS; se nunca
    # conseguir, esse UM segmento fica sem candidatos (dub.py cai pro texto
    # original em inglês) em vez de travar os outros 89. Só os jobs que
    # falharem são reenviados no próximo round — o lote encolhe a cada
    # tentativa, não refaz trabalho já bem-sucedido.
    MAX_TRANSLATE_ATTEMPTS = 3

    jobs = json.loads(Path(args.input).read_text(encoding="utf-8"))

    def build_prompt_text(job: dict) -> str:
        n_candidates = job.get("n_candidates", N_CANDIDATES)
        prompt = build_prompt(
            job["source_text"], job["target_syllables"], n_candidates=n_candidates
        )
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate_batch(batch_jobs: list[dict]) -> list[str]:
        texts = [build_prompt_text(job) for job in batch_jobs]
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
        outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.7)
        prompt_len = inputs["input_ids"].shape[1]
        return [
            tokenizer.decode(outputs[i][prompt_len:], skip_special_tokens=True)
            for i in range(len(batch_jobs))
        ]

    candidates_by_id: dict[str, list[str]] = {job["id"]: [] for job in jobs}
    pending = list(jobs)
    for attempt in range(1, MAX_TRANSLATE_ATTEMPTS + 1):
        if not pending:
            break
        still_pending = []
        for chunk_start in range(0, len(pending), TRANSLATE_BATCH_SIZE):
            chunk = pending[chunk_start : chunk_start + TRANSLATE_BATCH_SIZE]
            responses = generate_batch(chunk)
            for job, response in zip(chunk, responses):
                try:
                    candidates_by_id[job["id"]] = parse_candidates(response)
                except ValueError as exc:
                    print(
                        f"[translate] job {job['id']}: tentativa {attempt}/"
                        f"{MAX_TRANSLATE_ATTEMPTS} falhou ({exc})"
                    )
                    still_pending.append(job)
        pending = still_pending

    results = []
    for job in jobs:
        candidates = candidates_by_id[job["id"]]
        ranked = (
            rank_candidates(candidates, job["target_syllables"], job["source_text"])
            if candidates
            else []
        )
        results.append(
            {
                "id": job["id"],
                "candidates": [
                    {
                        "text": c.text,
                        "syllables": c.syllables,
                        "budget_score": c.budget_score,
                        "naturalness_score": c.naturalness_score,
                        "expressiveness_score": c.expressiveness_score,
                        "score": c.score,
                    }
                    for c in ranked
                ],
            }
        )

    Path(args.output).write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")


def cmd_synthesize(args: argparse.Namespace) -> None:
    import soundfile as sf
    import torch
    import torchaudio
    from transformers import AutoModel, AutoProcessor

    from packages.pipeline.segmentation import InternalPause
    from packages.pipeline.synthesis import (
        MIN_SYNTHESIS_SECONDS,
        adjust_tokens_for_retry,
        apply_ipa_overrides,
        duration_to_tokens,
        inject_pauses,
        is_within_tolerance,
        time_stretch_to_duration,
    )

    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    repo_id = "OpenMOSS-Team/MOSS-TTS-v1.5"
    device = "cuda"
    # Em GPUs de ~23GB (ex.: L4) o modelo base sozinho já usa ~21-22GB —
    # mover o audio_tokenizer pra GPU também estoura CUDA OOM por uma margem
    # mínima (~20MB, confirmado testando: nem device_map nem
    # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True resolvem, a folga
    # real não existe). Com o tokenizer na CPU o modelo cabe com folga
    # (~17GB), mas cada generate() paga um round-trip CPU↔GPU por token de
    # áudio decodificado, o que mede ~4-8s pra segmentos de 2-6s — acima do
    # invariante de <5s de CLAUDE.md pra ressíntese de segmento isolado.
    # Configurável por env porque é uma característica da GPU, não do
    # código: numa GPU com mais VRAM (A6000, A100...) "cuda" tem folga e é
    # bem mais rápido.
    audio_tokenizer_device = os.environ.get("MOSS_TTS_AUDIO_TOKENIZER_DEVICE", "cuda")
    language = "Portuguese"  # tende ao pt-PT sem essa tag explícita, ver docs/ptbr.md
    default_tolerance = 0.08

    processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)
    # device_map (em vez de carregar no CPU e mover com .to(device)) evita o
    # pico transitório de manter a cópia CPU e a GPU vivas ao mesmo tempo —
    # confirmado necessário numa L4 de 23GB: sem isso, mover o
    # audio_tokenizer pra GPU logo depois estourava CUDA OOM por ~20MB.
    model = AutoModel.from_pretrained(
        repo_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=device,
    )

    if args.lora_adapter_path:
        # Ver scripts/pod_lora_train.py pro porquê desses dois workarounds
        # (peft não foi feito pensando nesta classe de modelo). merge_and_unload
        # funde o adapter nos pesos base — necessário porque manter o wrapper
        # do peft ativo durante generate() deu CUBLAS_STATUS_EXECUTION_FAILED
        # (confirmado testando; funde-e-descarta não teve esse problema).
        from peft import PeftModel

        original_get_input_embeddings = type(model).get_input_embeddings
        type(model).get_input_embeddings = lambda self, input_ids=None: (
            original_get_input_embeddings(self, input_ids)
            if input_ids is not None
            else self.language_model.get_input_embeddings()
        )
        if not hasattr(type(model), "prepare_inputs_for_generation"):
            type(model).prepare_inputs_for_generation = lambda self, *a, **k: {}

        torch.cuda.empty_cache()
        model = PeftModel.from_pretrained(model, args.lora_adapter_path, is_trainable=False)
        model = model.merge_and_unload()

    # audio_tokenizer só entra na GPU depois do adapter fundido — carregar o
    # PeftModel sozinho já quase enche 24GB, confirmado na prática.
    torch.cuda.empty_cache()
    processor.audio_tokenizer = processor.audio_tokenizer.to(audio_tokenizer_device)

    # T0.14b — achado real (2026-09-15): gerar N amostras independentes da
    # MESMA frase numa única chamada em lote custa quase o mesmo tempo de
    # GPU que gerar 1 (confirmado: 3 amostras em 3,52s, 1 amostra sozinha
    # já levava 2-4s) — a instabilidade estocástica do MOSS-TTS (repetição/
    # alucinação) é melhor endereçada tendo várias tentativas PARALELAS
    # pra escolher a melhor depois (`scripts/dub.py::pick_best_synthesis`)
    # do que retentando em rodadas seriais inteiras (processo novo no Pod +
    # round-trip de avaliação por rodada — isso sim é caro, era o gargalo
    # real, não a geração em si). Substitui o retry por rodada do T0.12.
    #
    # Testado 5 em 2026-09-18 (beast.mp4) e revertido pra 3 no mesmo dia:
    # medido na prática que N=5 é mensuravelmente mais lento por chamada
    # (lote maior, ~20-24it/s vs ~24-25it/s em N=3) e o problema que N=5
    # tentava compensar (segmentos com duração completamente imprevisível)
    # era na real um bug de parsing na tradução (`translation.py::
    # parse_candidates`, corrigido no mesmo dia) mandando texto corrompido
    # pro MOSS-TTS — não instabilidade do modelo que mais amostra resolvia.
    # Com a causa raiz corrigida, N=3 já é suficiente; usuário pediu
    # explicitamente pra priorizar velocidade de prototipagem.
    N_CANDIDATES = 3

    # Achado real rodando beast.mp4 (2026-09-18): o bug de repetição do
    # MOSS-TTS já documentado (SGLang: "a small fraction of utterances loop
    # and generate up to max_new_tokens") apareceu ao vivo — 1 de 8 chamadas
    # em lote não parou sozinha e consumiu os 4096 tokens inteiros (~5,5min
    # só essa chamada, a ~12-13it/s), enquanto as outras 7 pararam sozinhas
    # entre 106 e 446 passos. Como a geração em lote só retorna quando TODAS
    # as sequências do batch terminam (ou batem o teto), 1 candidato solto em
    # loop trava os outros 2 que já tinham terminado. Não faz sentido deixar
    # um candidato sabidamente ruim (repetição/alucinação — nunca é o
    # escolhido por `pick_best_synthesis`) correr até 4096: cortar mais cedo
    # não piora a escolha final, só limita o desperdício. Teto escalado pelo
    # orçamento real do job (6x tokens, piso 1024) em vez de um número fixo —
    # cobre com folga o maior caso observado (446 passos pra ~150 tokens de
    # orçamento) sem arriscar cortar um segmento longo legítimo.
    MAX_NEW_TOKENS_MULTIPLIER = 6
    MIN_MAX_NEW_TOKENS = 1024

    def synthesize_batch(
        text: str,
        tokens: int,
        reference: list[str] | None,
        out_dir: Path,
        job_id: str,
        instruction: str | None,
    ) -> list[tuple[Path, float]]:
        message = processor.build_user_message(
            text=text,
            language=language,
            tokens=tokens,
            reference=reference,
            instruction=instruction,
        )
        batch = processor([[message]] * N_CANDIDATES, mode="generation")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        max_new_tokens = max(MIN_MAX_NEW_TOKENS, tokens * MAX_NEW_TOKENS_MULTIPLIER)
        outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens
        )

        sr = processor.model_config.sampling_rate
        results = []
        for i, decoded in enumerate(processor.decode(outputs)):
            audio = decoded.audio_codes_list[0]
            cand_path = out_dir / f"{job_id}_cand{i}.wav"
            torchaudio.save(str(cand_path), audio.unsqueeze(0), sr)
            results.append((cand_path, audio.shape[-1] / sr))
        return results

    def median_seconds(candidates: list[tuple[Path, float]]) -> float:
        seconds = sorted(secs for _, secs in candidates)
        return seconds[len(seconds) // 2] if seconds else 0.0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = json.loads(Path(args.input).read_text(encoding="utf-8"))
    meta = []
    for job in jobs:
        tolerance = job.get("tolerance", default_tolerance)
        pausas = [
            InternalPause(after_word_index=p["after_word_index"], duration=p["duration"])
            for p in job.get("pausas_internas", [])
        ]
        text = apply_ipa_overrides(job["translated_text"], job.get("glossary", {}))
        text = inject_pauses(text, pausas, job.get("original_word_count", 0))

        reference = [job["reference_audio_path"]] if job.get("reference_audio_path") else None
        target_seconds = job["target_seconds"]
        # DESLIGADO em 2026-09-19 — achado real, medido e reproduzido (não
        # suposição): `infer_delivery_instruction` era a causa raiz dos
        # piores casos de "MOSS-TTS instável" caçados a sessão inteira (ver
        # docs/moss_tts_investigation.md). Experimento controlado no texto
        # real de seg0011 (14 palavras + 1,8s de pausa em 3,2s de
        # orçamento, tokens=40): COM a instrução de ênfase, 10/10 amostras
        # entraram em loop (9-36s de áudio pra um alvo de 4,6s). SEM a
        # instrução, MESMO texto/tokens: 0/10 loop, todas as 10 amostras em
        # 2,88-3,44s — quase exatamente o orçamento pedido. O modelo
        # interpreta "fale com ênfase e emoção genuína" como licença pra
        # falar bem mais devagar/expansivo, e sob orçamento apertado isso
        # vira um pedido estruturalmente impossível — a instabilidade era
        # sintoma disso, não um bug aleatório do MOSS-TTS. Mantém a função
        # (testada, lógica de detecção de pontuação continua válida) mas
        # não chama mais aqui até haver uma versão redesenhada que não
        # comprometa o orçamento de duração.
        instruction = None

        tokens = duration_to_tokens(target_seconds)
        candidates = synthesize_batch(text, tokens, reference, out_dir, job["id"], instruction)
        attempts = 1

        if not is_within_tolerance(median_seconds(candidates), target_seconds, tolerance):
            tokens = adjust_tokens_for_retry(tokens, median_seconds(candidates), target_seconds)
            candidates = synthesize_batch(
                text, tokens, reference, out_dir, job["id"], instruction
            )
            attempts = 2

        within_tolerance = is_within_tolerance(
            median_seconds(candidates), target_seconds, tolerance
        )
        stretched = False

        # T0.15b, ampliado em 2026-09-18: achado real testando beast.mp4 —
        # um segmento bem ACIMA do piso (target 5,38s) saiu com 8,0s depois
        # do retry (49% acima do alvo, sem nenhuma correção), porque esta
        # condição só cobria o caso abaixo de MIN_SYNTHESIS_SECONDS. Isso
        # sobrou pra assembly.py, que estourou a timeline perto do fim do
        # vídeo (`TimelineOverflowError`, 2 segmentos somando 1,8s
        # descartado). O overshoot depois do retry não é exclusivo de
        # segmento no piso — qualquer segmento pode sair fora da tolerância
        # e precisar de compressão. Comprime sempre que sobrar fora da
        # tolerância depois do retry, não só no caso do piso.
        if not within_tolerance:
            stretched_candidates = []
            for cand_path, _secs in candidates:
                audio, sample_rate = sf.read(str(cand_path), dtype="float32")
                audio = time_stretch_to_duration(audio, sample_rate, target_seconds)
                sf.write(str(cand_path), audio, sample_rate)
                stretched_candidates.append((cand_path, len(audio) / sample_rate))
            candidates = stretched_candidates
            within_tolerance = is_within_tolerance(
                median_seconds(candidates), target_seconds, tolerance
            )
            stretched = True

        # Achado real 2026-09-18 (beast.mp4, N=3): pra segmentos JÁ acima do
        # piso de síntese, um resultado ainda gritantemente fora mesmo
        # depois do retry+stretch (visto: alvo 4,58s, os 3 candidatos
        # saíram 11,5-52,9s — bug de repetição documentado do MOSS-TTS, não
        # o piso estrutural) não tem conserto por compressão: o conteúdo em
        # si está corrompido (loop), não só mais longo que devia. Gerar
        # candidatos extras SÓ nesse caso raro (em vez de sempre pedir mais
        # amostras por segurança, caro pra todo segmento) dá uma chance real
        # de uma amostra nova escapar do loop, sem pagar esse custo nos
        # ~90% dos segmentos que já funcionam de primeira. Não aplica pra
        # segmento abaixo do piso (`target_seconds < MIN_SYNTHESIS_SECONDS`)
        # — ali o "excesso" é estrutural e esperado (ver
        # assembly.MAX_SINGLE_SEGMENT_DISCARD_SECONDS), mais amostra não
        # muda o piso físico do modelo.
        ESCALATION_RATIO_THRESHOLD = 1.5
        escalated = False
        if not within_tolerance and target_seconds >= MIN_SYNTHESIS_SECONDS:
            best_ratio_error = min(
                abs(secs / target_seconds - 1) for _, secs in candidates
            )
            if best_ratio_error > (ESCALATION_RATIO_THRESHOLD - 1):
                # Mesmo N_CANDIDATES de sempre (lote extra, não é mais caro
                # por amostra do que qualquer outra chamada) — só o GATILHO
                # é raro, não o tamanho do lote extra.
                extra_raw = synthesize_batch(
                    text, tokens, reference, out_dir, f"{job['id']}_esc", instruction
                )
                extra: list[tuple[Path, float]] = []
                for offset, (extra_path, secs) in enumerate(extra_raw):
                    final_path = out_dir / f"{job['id']}_cand{len(candidates) + offset}.wav"
                    extra_path.rename(final_path)
                    extra.append((final_path, secs))
                candidates = candidates + extra
                within_tolerance = is_within_tolerance(
                    median_seconds(candidates), target_seconds, tolerance
                )
                escalated = True

        meta.append(
            {
                "id": job["id"],
                "candidates": [
                    {"path": cand_path.name, "achieved_seconds": secs}
                    for cand_path, secs in candidates
                ],
                "achieved_seconds": median_seconds(candidates),
                "tokens_used": tokens,
                "attempts": attempts,
                "within_tolerance": within_tolerance,
                "stretched": stretched,
                "escalated": escalated,
            }
        )

    (out_dir / "synth_meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def cmd_evaluate(args: argparse.Namespace) -> None:
    from faster_whisper import WhisperModel

    model = WhisperModel("large-v3", device="cuda", compute_type="float16")

    jobs = json.loads(Path(args.input).read_text(encoding="utf-8"))
    results = []
    for job in jobs:
        # condition_on_previous_text=True (o default) alucina texto repetido
        # em loop em áudio curto — confirmado direto num áudio real de 4,2s,
        # uma frase só, que sem esse ajuste virava a mesma frase 16x seguidas.
        # Bug conhecido do faster-whisper, não do áudio sendo avaliado.
        #
        # T0.17 achado real: mesmo com o ajuste acima, o Whisper AINDA
        # alucina frase repetida em alguns clipes curtos — confirmado
        # contando rajadas de energia no áudio (5-6 rajadas reais, texto
        # dizendo a mesma frase de 5-6 palavras 28x). Isso inflava muito a
        # taxa de "síntese ruim"/retry sem o MOSS-TTS ter feito nada de
        # errado. `repetition_penalty`/`no_repeat_ngram_size` (mesma técnica
        # de anti-loop de LLM de texto, aplicada aqui no decoder do Whisper)
        # resolveu 2 de 3 casos reais testados por completo; o 3º melhorou
        # bastante mas não 100% — ver docs/known_issues se persistir.
        segments, _info = model.transcribe(
            job["audio_path"],
            language="pt",
            beam_size=5,
            condition_on_previous_text=False,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
        )
        text = " ".join(segment.text.strip() for segment in segments)
        results.append({"id": job["id"], "text": text})

    Path(args.output).write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")


def cmd_match_emotion(args: argparse.Namespace) -> None:
    """Compara a emoção do áudio fonte com a do áudio dublado (item 4 do
    plano de avaliação de qualidade, 2026-09-26 — usuário pediu detecção
    automática de emoção, não só ouvido).

    `emotion2vec_plus_large` (ACL 2024, via FunASR) classifica um trecho de
    áudio num vetor de scores por classe (raiva, alegria, neutro etc.) —
    roda uma vez por chamada de processo (mesmo padrão de `cmd_evaluate`),
    carrega o modelo uma vez, processa o lote inteiro. A comparação em si
    (similaridade de cosseno + limiar) é pura e mora em
    `packages.pipeline.quality` — este comando só extrai o vetor do modelo
    e delega o julgamento pra lá, mesma separação usada em `entonacao_desalinhada`.

    Cada job: `id`, `dub_audio_path` (WAV do segmento já sintetizado, um
    arquivo por segmento — mesmo layout de `cmd_evaluate`), `source_audio_path`
    (o vocals.wav CHEIO do vídeo original — um só arquivo, compartilhado
    entre todos os segmentos), `source_start`/`source_end` (segundos, pra
    recortar o trecho correspondente do vocals.wav aqui dentro). Evita subir
    um WAV por segmento da fonte só pra isso — o vocals.wav inteiro já
    existe no Pod desde a separação de stems.
    """
    import numpy as np
    import soundfile as sf
    from funasr import AutoModel

    from packages.pipeline.quality import emocao_incompativel, emotion_similarity

    model = AutoModel(model="iic/emotion2vec_plus_large", disable_update=True)

    def _classify(path: str) -> tuple[list[str], list[float]]:
        res = model.generate(path, granularity="utterance", extract_embedding=False)
        labels = [label.rsplit("/", 1)[-1] for label in res[0]["labels"]]
        return labels, res[0]["scores"]

    tmp_dir = Path("/tmp/novoaudio_emotion")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    source_cache: dict[str, tuple[np.ndarray, int]] = {}

    jobs = json.loads(Path(args.input).read_text(encoding="utf-8"))
    results = []
    for job in jobs:
        source_path = job["source_audio_path"]
        if source_path not in source_cache:
            audio, sr = sf.read(source_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            source_cache[source_path] = (audio, sr)
        source_audio, source_sr = source_cache[source_path]

        start_sample = int(round(job["source_start"] * source_sr))
        end_sample = int(round(job["source_end"] * source_sr))
        source_slice_path = tmp_dir / f"{job['id']}_source.wav"
        sf.write(source_slice_path, source_audio[start_sample:end_sample], source_sr)

        source_labels, source_scores = _classify(str(source_slice_path))
        dub_labels, dub_scores = _classify(job["dub_audio_path"])
        similarity = emotion_similarity(np.array(source_scores), np.array(dub_scores))
        source_top = int(np.argmax(source_scores))
        dub_top = int(np.argmax(dub_scores))
        results.append(
            {
                "id": job["id"],
                "similarity": similarity,
                "emocao_incompativel": emocao_incompativel(similarity),
                "source_emotion": source_labels[source_top],
                "source_emotion_score": source_scores[source_top],
                "dub_emotion": dub_labels[dub_top],
                "dub_emotion_score": dub_scores[dub_top],
            }
        )

    Path(args.output).write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    p = subparsers.add_parser("separate_stems")
    p.add_argument("--input", required=True)
    p.add_argument("--out-dir", required=True)
    p.set_defaults(func=cmd_separate_stems)

    p = subparsers.add_parser("transcribe")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_transcribe)

    p = subparsers.add_parser("translate")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_translate)

    p = subparsers.add_parser("synthesize")
    p.add_argument("--input", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--lora-adapter-path",
        default=None,
        help="caminho (no Pod) de um adapter LoRA já treinado; funde nos pesos base antes de gerar",
    )
    p.set_defaults(func=cmd_synthesize)

    p = subparsers.add_parser("evaluate")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_evaluate)

    p = subparsers.add_parser("match_emotion")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_match_emotion)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
