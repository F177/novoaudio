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


def _patch_torchaudio_backend_shim() -> None:
    """pyannote.audio==3.1.1 (pinado transitivamente pelo whisperx 3.2.0) usa a
    API antiga de backend do torchaudio, removida em versões novas. Chamar
    antes de qualquer `import whisperx` (mesmo shim de `runpod/transcribe/_compat.py`,
    duplicado aqui pra este script não depender de um pacote local chamado
    `runpod`, que colidiria com o SDK `runpod` instalado via pip).
    """
    import torchaudio

    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: "soundfile"
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *args, **kwargs: None
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]


def cmd_transcribe(args: argparse.Namespace) -> None:
    _patch_torchaudio_backend_shim()
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
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device
    )

    # Qwen às vezes não devolve JSON válido (achado real, 2x em bateladas de
    # 90+ segmentos) — sem retry, 1 job ruim derrubava a tradução inteira do
    # vídeo, mesmo com todos os outros já traduzidos certo. Reamostra (a
    # geração é `do_sample=True`, então tentar de novo é uma chance real de
    # sucesso, não repetir o mesmo erro) até MAX_TRANSLATE_ATTEMPTS; se nunca
    # conseguir, esse UM segmento fica sem candidatos (dub.py cai pro texto
    # original em inglês) em vez de travar os outros 89.
    MAX_TRANSLATE_ATTEMPTS = 3

    jobs = json.loads(Path(args.input).read_text(encoding="utf-8"))
    results = []
    for job in jobs:
        n_candidates = job.get("n_candidates", N_CANDIDATES)
        prompt = build_prompt(
            job["source_text"], job["target_syllables"], n_candidates=n_candidates
        )
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(device)

        candidates: list[str] = []
        for attempt in range(1, MAX_TRANSLATE_ATTEMPTS + 1):
            outputs = model.generate(
                **inputs, max_new_tokens=1024, do_sample=True, temperature=0.7
            )
            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            try:
                candidates = parse_candidates(response)
                break
            except ValueError as exc:
                print(
                    f"[translate] job {job['id']}: tentativa {attempt}/"
                    f"{MAX_TRANSLATE_ATTEMPTS} falhou ({exc})"
                )

        ranked = rank_candidates(candidates, job["target_syllables"]) if candidates else []
        results.append(
            {
                "id": job["id"],
                "candidates": [
                    {
                        "text": c.text,
                        "syllables": c.syllables,
                        "budget_score": c.budget_score,
                        "naturalness_score": c.naturalness_score,
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

    def synthesize_once(text: str, tokens: int, reference: list[str] | None, out_path: Path):
        message = processor.build_user_message(
            text=text, language=language, tokens=tokens, reference=reference
        )
        batch = processor([[message]], mode="generation")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=4096
        )

        achieved_seconds = 0.0
        for decoded in processor.decode(outputs):
            audio = decoded.audio_codes_list[0]
            sr = processor.model_config.sampling_rate
            torchaudio.save(str(out_path), audio.unsqueeze(0), sr)
            achieved_seconds = audio.shape[-1] / sr
            break
        return achieved_seconds

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
        out_path = out_dir / f"{job['id']}.wav"

        tokens = duration_to_tokens(target_seconds)
        achieved_seconds = synthesize_once(text, tokens, reference, out_path)
        attempts = 1

        if not is_within_tolerance(achieved_seconds, target_seconds, tolerance):
            tokens = adjust_tokens_for_retry(tokens, achieved_seconds, target_seconds)
            achieved_seconds = synthesize_once(text, tokens, reference, out_path)
            attempts = 2

        within_tolerance = is_within_tolerance(achieved_seconds, target_seconds, tolerance)
        stretched = False

        # T0.15b: segmento abaixo do piso do delay pattern (MIN_SYNTHESIS_
        # SECONDS) sai sistematicamente mais longo que o alvo real, porque
        # duration_to_tokens/adjust_tokens_for_retry nunca pedem menos que
        # o piso (ver docstring de synthesis.py). Comprime de volta em vez
        # de deixar sobrar pra assembly.py estourar a timeline (T0.16) ou
        # tocar mais devagar que devia.
        if not within_tolerance and target_seconds < MIN_SYNTHESIS_SECONDS:
            audio, sample_rate = sf.read(str(out_path), dtype="float32")
            audio = time_stretch_to_duration(audio, sample_rate, target_seconds)
            sf.write(str(out_path), audio, sample_rate)
            achieved_seconds = len(audio) / sample_rate
            within_tolerance = is_within_tolerance(achieved_seconds, target_seconds, tolerance)
            stretched = True

        meta.append(
            {
                "id": job["id"],
                "achieved_seconds": achieved_seconds,
                "tokens_used": tokens,
                "attempts": attempts,
                "within_tolerance": within_tolerance,
                "stretched": stretched,
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

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
