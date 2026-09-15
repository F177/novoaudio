"""CLI de ponta a ponta (T0.12): vídeo em inglês entra, vídeo dublado em
pt-BR sai, com relatório JSON de segmentos e flags.

Orquestra as etapas de GPU (Demucs, WhisperX, Qwen2.5-7B-Instruct,
MOSS-TTS, Whisper large-v3) rodando remotamente num Pod RunPod comum via
SSH — ver `scripts/pod_worker.py` (o que roda no Pod) e
`scripts/pod_runner.py` (como este script fala com o Pod). Não usa
serverless/R2 ainda (decisão do dono do produto: validar a hipótese do
pipeline antes de rodar contra os 5 endpoints, ver `runpod/README.md`).
As etapas puras (segmentação, orçamento de duração, montagem, gates de
qualidade) rodam localmente via `packages/pipeline`.

Cacheia cada etapa em disco (`--cache-dir`, default `.cache/dub/<video>/`)
e pula qualquer etapa cujo arquivo de saída já exista — `--force` ignora
o cache e refaz tudo. Isso é o que torna o CLI seguro de re-rodar depois
de uma falha de rede/GPU no meio do vídeo, sem pagar de novo pelas etapas
que já tinham funcionado.

**Simplificações conhecidas desta versão** (documentadas, não escondidas):
- Não há detecção de "locutor em quadro" (isso exigiria visão computacional,
  não construído ainda) — todo segmento usa a tolerância mais larga
  (±20%, off-screen) até essa detecção existir.
- `traducao_infiel` e `locutor_suspeito` (T0.11) não são calculadas de
  verdade aqui: exigiriam, respectivamente, um score de similaridade
  semântica fonte/tradução e um modelo de similaridade de locutor
  (WavLM-TDNN/ECAPA) — nenhum dos dois foi implementado como worker ainda.
  Ambas ficam `False` no relatório, marcadas com `"not_computed": true`.
- Escolhe sempre o candidato de tradução melhor-pontuado (`rank_candidates`
  já ordena) — não há edição manual de segmento neste CLI (isso é o
  editor web, Fase 1).
- Entrada sem faixa de vídeo (ex.: mp3/m4a) é aceita: pula o remux final e
  entrega só o áudio dublado (`.wav`) — útil pra validar a qualidade do
  pipeline antes de ter vídeo de teste de verdade.

Uso (rodar como módulo, `-m`, não `python scripts/dub.py` — senão
`packages`/`scripts` não entram no sys.path e o import falha):
    python -m scripts.dub video.mp4
    python -m scripts.dub video.mp4 --out saida.mp4 --cache-dir .cache/dub --force
    python -m scripts.dub --r2-key "pasta/video.mp4"   # baixa do bucket R2 antes
    make pipeline-cli VIDEO=path/to.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import soundfile as sf
from dotenv import load_dotenv

from packages.pipeline.assembly import (
    TimedAudio,
    mix_with_background,
    normalize_loudness,
    place_segments_on_timeline,
    remux_with_video,
)
from packages.pipeline.budget import TOLERANCE_OFF_SCREEN, load_calibration, syllables_that_fit
from packages.pipeline.quality import (
    QualityFlags,
    cer_alto,
    character_error_rate,
    clipping,
    fora_duracao,
    has_repetition,
    silencio_anormal,
    truncado,
)
from packages.pipeline.segmentation import Segment, Word, segment_words
from packages.pipeline.synthesis import is_within_tolerance
from scripts import r2
from scripts.pod_runner import PodConfig, download_from_pod, run_on_pod, upload_to_pod

SYNTH_SAMPLE_RATE = 24000  # MOSS-TTS-v1.5, ver packages/pipeline/synthesis.py


def _is_bad_synthesis(reference_text: str, hypothesis_text: str) -> bool:
    """Heurística de "essa síntese saiu ruim, vale tentar de novo".

    T0.14 — achado real: o predicado antigo (`has_repetition or CER >= 1.0`)
    só pegava falha catastrófica (loop óbvio ou transcrição vazia/lixo
    total). Um segmento com o final cortado pelo MOSS-TTS costuma sair com
    CER moderado (~0,1-0,2 numa frase longa, ver docstring de
    `character_error_rate`) — nunca batia `>= 1.0`, e passava incólume pro
    relatório final. Agora usa `cer_alto` (limiar real do T0.13, medido) e
    `truncado` (lexical, pega corte que `fora_duracao`/CER não pegam)."""
    cer = character_error_rate(reference_text, hypothesis_text)
    return (
        has_repetition(hypothesis_text)
        or truncado(reference_text, hypothesis_text)
        or cer_alto(cer)
    )


def pick_best_synthesis(reference_text: str, hypotheses: list[str]) -> int:
    """Escolhe o índice do melhor candidato entre `hypotheses` — N amostras
    independentes do MESMO segmento, geradas numa única chamada em lote
    (ver `scripts/pod_worker.py::cmd_synthesize`).

    Achado real (sessão de 2026-09-14/15): o retry sequencial antigo (gerar
    de novo em rodadas separadas, cada uma um processo novo no Pod +
    round-trip de avaliação) custava minutos por rodada — quase todo esse
    custo é overhead de processo/rede, não de GPU (3 amostras em lote saem
    em ~3,5s, quase o mesmo tempo de 1 amostra sozinha, confirmado testando
    na prática). Gerar N amostras de uma vez e escolher a melhor localmente
    substitui o retry por rodadas inteiras, com o mesmo espírito do T0.14
    (`_is_bad_synthesis`) mas sem o custo.

    Primeiro candidato que não é "ruim" vence; se nenhum servir, o de
    menor CER (ainda assim o "menos pior", não necessariamente bom —
    `_is_bad_synthesis` continua rodando sobre o vencedor depois, pra
    reportar a flag certa).
    """
    if not hypotheses:
        return 0
    for i, hyp in enumerate(hypotheses):
        if not _is_bad_synthesis(reference_text, hyp):
            return i
    return min(
        range(len(hypotheses)),
        key=lambda i: character_error_rate(reference_text, hypotheses[i]),
    )


def _sanitize_for_dirname(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _has_video_stream(path: Path) -> bool:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v",
            "-show_entries", "stream=index", "-of", "csv=p=0", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    return bool(result.stdout.strip())


def words_from_whisperx_transcript(transcript: dict[str, Any]) -> list[Word]:
    """Achata `result["segments"][*]["words"]` do WhisperX numa lista única de
    `Word`, ordenada por tempo. Palavras sem timestamp (falha de alinhamento)
    são descartadas — não têm posição pra entrar numa segmentação por pausa.
    """
    words: list[Word] = []
    for seg in transcript.get("segments", []):
        for w in seg.get("words", []):
            if "start" not in w or "end" not in w:
                continue
            words.append(
                Word(text=w["word"], start=w["start"], end=w["end"], speaker=w.get("speaker"))
            )
    words.sort(key=lambda w: w.start)
    return words


def build_stage_report(segment_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Taxa de sucesso por etapa (critério de aceite do T0.12) e taxa de
    segmentos com pelo menos uma flag (usada no portão de decisão de
    Fase 0: >30% precisando de correção manual).
    """
    total = len(segment_results)
    if total == 0:
        return {"total_segments": 0}

    def rate(key: str) -> float:
        return sum(1 for r in segment_results if r.get(key)) / total

    needed_retry = sum(1 for r in segment_results if r.get("synthesis_attempts", 1) > 1)
    # T0.14b: quantas vezes o 1º candidato do lote não servia e precisou
    # escolher outro (proxy de instabilidade estocástica do MOSS-TTS que o
    # retry por lote absorve — antes disso, cada um desses seria uma rodada
    # inteira de retry, cara; agora é análise local, praticamente de graça).
    needed_non_first_candidate = sum(
        1 for r in segment_results if r.get("chosen_candidate_index", 0) > 0
    )

    return {
        "total_segments": total,
        "translated_rate": rate("translated_ok"),
        "synthesized_rate": rate("synthesized_ok"),
        "within_duration_tolerance_rate": rate("within_duration_tolerance"),
        "evaluated_rate": rate("evaluated_ok"),
        "segments_with_any_flag_rate": rate("any_flag"),
        "segments_needing_synth_retry_rate": needed_retry / total,
        "segments_needing_non_first_candidate_rate": needed_non_first_candidate / total,
    }


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _probe_duration_seconds(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def _extract_audio(video_path: Path, out_path: Path) -> None:
    _run(["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ar", "44100", "-ac", "2", str(out_path)])


def _resample(in_path: Path, out_path: Path, sample_rate: int) -> None:
    _run(["ffmpeg", "-y", "-i", str(in_path), "-ar", str(sample_rate), "-ac", "1", str(out_path)])


def save_transcript_and_translation(
    report: dict[str, Any], cache_dir: Path, out_path: Path
) -> tuple[Path, Path]:
    """Salva o transcript original (WhisperX) e a tradução REALMENTE usada
    na síntese (não os 5 candidatos — só o escolhido, por segmento) ao lado
    do vídeo de saída, pra inspeção sem precisar abrir `--cache-dir`.

    O transcript salvo aqui é simplificado (`start, end, text` por
    segmento) — o WhisperX bruto em `--cache-dir/transcript.json` vem numa
    linha só e com o detalhe por palavra (score, speaker), que é ruído pra
    quem só quer conferir o que foi dito. Indentado (`indent=2`) pelos
    mesmos motivos.

    Devolve os dois caminhos escritos (transcript, tradução), mesmo que o
    transcript não exista (ex.: reused de um cache antigo sem essa etapa).
    """
    transcript_dest = out_path.with_name(f"{out_path.stem}_transcript.json")
    transcript_src = cache_dir / "transcript.json"
    if transcript_src.exists():
        raw_transcript = json.loads(transcript_src.read_text(encoding="utf-8"))
        simplified = [
            {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
            for seg in raw_transcript.get("segments", [])
        ]
        transcript_dest.write_text(
            json.dumps(simplified, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    translation_rows = [
        {
            "id": s["id"],
            "t_inicio": s["t_inicio"],
            "t_fim": s["t_fim"],
            "texto_original": s.get("texto_original"),
            "traducao_usada": s.get("traducao"),
        }
        for s in report["segments"]
    ]
    translation_dest = out_path.with_name(f"{out_path.stem}_traducao.json")
    translation_dest.write_text(
        json.dumps(translation_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return transcript_dest, translation_dest


def run_pipeline(
    video_path: Path,
    out_path: Path,
    cache_dir: Path,
    force: bool,
    *,
    vocals_only: bool = False,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    pod = PodConfig.from_env()
    remote_job_dir = f"jobs/{cache_dir.name}"
    run_on_pod(pod, f"mkdir -p {remote_job_dir}")
    hf_env = {"HF_HOME": "/workspace/hf_cache", "HF_TOKEN": os.environ.get("HF_TOKEN", "")}
    synth_env = {**hf_env, "MOSS_TTS_AUDIO_TOKENIZER_DEVICE": pod.audio_tokenizer_device}
    lora_flag = (
        f" --lora-adapter-path {pod.lora_adapter_path}" if pod.lora_adapter_path else ""
    )

    def cached(name: str) -> Path:
        return cache_dir / name

    def needs(path: Path) -> bool:
        return force or not path.exists()

    # --- 1. extrai áudio original (local, ffmpeg, sem GPU) ---
    audio_path = cached("audio.wav")
    if needs(audio_path):
        _extract_audio(video_path, audio_path)

    # --- 2. separação de stems (remoto, Demucs) ---
    vocals_remote = f"{remote_job_dir}/stems/vocals.wav"
    background_path = cached("background.wav")
    if needs(background_path):
        upload_to_pod(pod, str(audio_path), f"{remote_job_dir}/audio.wav")
        run_on_pod(
            pod,
            f"{pod.python_asr} -m scripts.pod_worker separate_stems "
            f"--input {remote_job_dir}/audio.wav --out-dir {remote_job_dir}/stems",
        )
        download_from_pod(pod, f"{remote_job_dir}/stems/background.wav", str(background_path))

    background_resampled_path = cached("background_24k.wav")
    if needs(background_resampled_path):
        _resample(background_path, background_resampled_path, SYNTH_SAMPLE_RATE)

    # --- 3. transcrição + diarização (remoto, WhisperX) ---
    transcript_path = cached("transcript.json")
    if needs(transcript_path):
        run_on_pod(
            pod,
            f"{pod.python_asr} -m scripts.pod_worker transcribe "
            f"--input {vocals_remote} --output {remote_job_dir}/transcript.json",
            env=hf_env,
        )
        download_from_pod(pod, f"{remote_job_dir}/transcript.json", str(transcript_path))

    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))

    # --- 4. segmentação prosódica (local, pura) ---
    segments_path = cached("segments.json")
    words = words_from_whisperx_transcript(transcript)
    segments: list[Segment] = segment_words(words)
    if needs(segments_path):
        segments_path.write_text(
            json.dumps(
                [
                    {
                        "id": f"seg{i:04d}",
                        "t_inicio": s.t_inicio,
                        "t_fim": s.t_fim,
                        "texto": s.texto,
                        "speaker": s.speaker,
                        "pausas_internas": [
                            {"after_word_index": p.after_word_index, "duration": p.duration}
                            for p in s.pausas_internas
                        ],
                    }
                    for i, s in enumerate(segments)
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    calibration = load_calibration()

    # --- 5. tradução com restrição de sílabas (remoto, Qwen2.5) ---
    translations_path = cached("translations.json")
    if needs(translations_path):
        translate_jobs = []
        for i, s in enumerate(segments):
            pause_seconds = sum(p.duration for p in s.pausas_internas)
            target_seconds = s.t_fim - s.t_inicio
            target_syllables = syllables_that_fit(
                target_seconds, calibration, pause_seconds=pause_seconds
            )
            translate_jobs.append(
                {
                    "id": f"seg{i:04d}",
                    "source_text": s.texto,
                    "target_syllables": max(1, target_syllables),
                }
            )
        (cache_dir / "translate_jobs.json").write_text(
            json.dumps(translate_jobs, ensure_ascii=False), encoding="utf-8"
        )
        upload_to_pod(
            pod, str(cache_dir / "translate_jobs.json"), f"{remote_job_dir}/translate_jobs.json"
        )
        run_on_pod(
            pod,
            f"{pod.python_asr} -m scripts.pod_worker translate "
            f"--input {remote_job_dir}/translate_jobs.json "
            f"--output {remote_job_dir}/translations.json",
            env=hf_env,
        )
        download_from_pod(pod, f"{remote_job_dir}/translations.json", str(translations_path))

    translations = {
        t["id"]: t["candidates"]
        for t in json.loads(translations_path.read_text(encoding="utf-8"))
    }

    segments_by_id = {f"seg{i:04d}": s for i, s in enumerate(segments)}
    best_text_by_id = {}
    for seg_id, s in segments_by_id.items():
        candidates = translations.get(seg_id, [])
        best_text_by_id[seg_id] = candidates[0]["text"] if candidates else s.texto

    def build_synth_job(seg_id: str) -> dict[str, Any]:
        s = segments_by_id[seg_id]
        return {
            "id": seg_id,
            "translated_text": best_text_by_id[seg_id],
            "target_seconds": s.t_fim - s.t_inicio,
            "pausas_internas": [
                {"after_word_index": p.after_word_index, "duration": p.duration}
                for p in s.pausas_internas
            ],
            "original_word_count": len(s.texto.split()),
            "tolerance": TOLERANCE_OFF_SCREEN,
            # Sem isso, o MOSS-TTS não ancora nenhuma voz e sorteia um timbre
            # novo por chamada (achado real ouvindo o T0.12: voz mudava a
            # cada segmento). Ver docstring de PodConfig.
            "reference_audio_path": pod.voice_reference,
        }

    # --- 6. síntese com duração e pausa (remoto, MOSS-TTS) ---
    # T0.14b: cada job gera N candidatos independentes numa única chamada em
    # lote (`pod_worker.py::cmd_synthesize`) — substitui o retry por rodadas
    # inteiras do T0.12/T0.14 (caro: processo novo no Pod + round-trip de
    # avaliação por rodada) por escolher o melhor entre N amostras já
    # geradas, localmente, sem round-trip extra de rede/GPU.
    synth_dir = cached("synth")
    synth_meta_path = synth_dir / "synth_meta.json"
    if needs(synth_meta_path):
        synth_jobs = [build_synth_job(seg_id) for seg_id in segments_by_id]
        (cache_dir / "synth_jobs.json").write_text(
            json.dumps(synth_jobs, ensure_ascii=False), encoding="utf-8"
        )
        upload_to_pod(pod, str(cache_dir / "synth_jobs.json"), f"{remote_job_dir}/synth_jobs.json")
        run_on_pod(
            pod,
            f"{pod.python_tts} -m scripts.pod_worker synthesize "
            f"--input {remote_job_dir}/synth_jobs.json --out-dir {remote_job_dir}/synth"
            f"{lora_flag}",
            env=synth_env,
        )
        download_from_pod(pod, f"{remote_job_dir}/synth", str(synth_dir))

    synth_meta = {m["id"]: m for m in json.loads(synth_meta_path.read_text(encoding="utf-8"))}

    # --- 7. avaliação CER round-trip (remoto, Whisper large-v3) — um job por candidato ---
    eval_path = cached("eval.json")
    if needs(eval_path):
        eval_jobs = [
            {"id": f"{seg_id}_cand{i}", "audio_path": f"{remote_job_dir}/synth/{cand['path']}"}
            for seg_id, meta in synth_meta.items()
            for i, cand in enumerate(meta["candidates"])
        ]
        (cache_dir / "eval_jobs.json").write_text(
            json.dumps(eval_jobs, ensure_ascii=False), encoding="utf-8"
        )
        upload_to_pod(pod, str(cache_dir / "eval_jobs.json"), f"{remote_job_dir}/eval_jobs.json")
        run_on_pod(
            pod,
            f"{pod.python_asr} -m scripts.pod_worker evaluate "
            f"--input {remote_job_dir}/eval_jobs.json --output {remote_job_dir}/eval.json",
            env={**hf_env, "LD_LIBRARY_PATH": pod.ld_library_path_asr},
        )
        download_from_pod(pod, f"{remote_job_dir}/eval.json", str(eval_path))

    candidate_eval_texts = {
        e["id"]: e["text"] for e in json.loads(eval_path.read_text(encoding="utf-8"))
    }

    # --- escolhe o melhor candidato por segmento, localmente, sem round-trip ---
    eval_texts: dict[str, str] = {}
    synth_attempts: dict[str, int] = {}
    for seg_id, meta in synth_meta.items():
        candidates = meta["candidates"]
        hypotheses = [
            candidate_eval_texts.get(f"{seg_id}_cand{i}", "") for i in range(len(candidates))
        ]
        winner = pick_best_synthesis(best_text_by_id.get(seg_id, ""), hypotheses)
        chosen = candidates[winner]

        chosen_path = synth_dir / chosen["path"]
        final_path = synth_dir / f"{seg_id}.wav"
        if chosen_path != final_path:
            final_path.write_bytes(chosen_path.read_bytes())

        eval_texts[seg_id] = hypotheses[winner] if hypotheses else ""
        synth_attempts[seg_id] = meta["attempts"]
        meta["achieved_seconds"] = chosen["achieved_seconds"]
        meta["within_tolerance"] = is_within_tolerance(
            chosen["achieved_seconds"],
            segments_by_id[seg_id].t_fim - segments_by_id[seg_id].t_inicio,
            TOLERANCE_OFF_SCREEN,
        )
        meta["chosen_candidate_index"] = winner

    # --- 8. gates de qualidade + montagem (local) ---
    background_audio, bg_sr = sf.read(background_resampled_path, dtype="float32")
    if background_audio.ndim > 1:
        background_audio = background_audio.mean(axis=1)

    timed_segments: list[TimedAudio] = []
    segment_results: list[dict[str, Any]] = []

    for i, s in enumerate(segments):
        seg_id = f"seg{i:04d}"
        target_seconds = s.t_fim - s.t_inicio
        translated_text = best_text_by_id[seg_id]
        translated_ok = bool(translations.get(seg_id))

        meta = synth_meta.get(seg_id)
        synthesized_ok = meta is not None
        wav_path = synth_dir / f"{seg_id}.wav"

        result: dict[str, Any] = {
            "id": seg_id,
            "t_inicio": s.t_inicio,
            "t_fim": s.t_fim,
            "texto_original": s.texto,
            "traducao": translated_text,
            "translated_ok": translated_ok,
            "synthesized_ok": synthesized_ok,
            "synthesis_attempts": synth_attempts.get(seg_id, 1),
            "chosen_candidate_index": meta.get("chosen_candidate_index", 0) if meta else 0,
        }

        if synthesized_ok and wav_path.exists():
            audio, sr = sf.read(wav_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            timed_segments.append(TimedAudio(start_seconds=s.t_inicio, audio=audio))

            achieved_seconds = meta["achieved_seconds"]
            expected_pause_seconds = sum(p.duration for p in s.pausas_internas)
            hypothesis = eval_texts.get(seg_id, "")
            evaluated_ok = seg_id in eval_texts
            cer = character_error_rate(translated_text, hypothesis) if evaluated_ok else 1.0

            flags = QualityFlags(
                fora_duracao=fora_duracao(achieved_seconds, target_seconds, on_screen=False),
                cer_alto=cer_alto(cer),
                truncado=truncado(translated_text, hypothesis) if evaluated_ok else False,
                traducao_infiel=False,
                clipping=clipping(audio),
                silencio_anormal=silencio_anormal(audio, sr, expected_pause_seconds),
                locutor_suspeito=False,
            )
            result.update(
                {
                    "achieved_seconds": achieved_seconds,
                    "target_seconds": target_seconds,
                    "within_duration_tolerance": meta["within_tolerance"],
                    "evaluated_ok": evaluated_ok,
                    "cer": cer,
                    "flags": {
                        **{k: v for k, v in vars(flags).items()},
                        "traducao_infiel_not_computed": True,
                        "locutor_suspeito_not_computed": True,
                    },
                    "any_flag": any(vars(flags).values()),
                }
            )
        else:
            result.update(
                {
                    "within_duration_tolerance": False,
                    "evaluated_ok": False,
                    "any_flag": True,
                }
            )

        segment_results.append(result)

    total_duration = _probe_duration_seconds(video_path)
    placement = place_segments_on_timeline(timed_segments, total_duration, SYNTH_SAMPLE_RATE)
    if vocals_only:
        # --vocals-only (T0.12, pedido pontual pra avaliar a voz sem o fundo
        # original atrapalhando) — comportamento padrão do produto continua
        # remixando com o fundo (ver mix_with_background abaixo).
        mixed_audio = placement.timeline
        mix_discarded_samples = 0
    else:
        mix = mix_with_background(placement.timeline, background_audio, SYNTH_SAMPLE_RATE)
        mixed_audio = mix.audio
        mix_discarded_samples = mix.discarded_samples
    normalized = normalize_loudness(mixed_audio, SYNTH_SAMPLE_RATE)

    final_audio_path = cached("final_audio.wav")
    sf.write(final_audio_path, normalized, SYNTH_SAMPLE_RATE)

    if _has_video_stream(video_path):
        remux_with_video(video_path, final_audio_path, out_path)
    else:
        # entrada sem faixa de vídeo (ex.: mp3/m4a) — não há o que remuxar,
        # entrega o áudio dublado direto (ver docstring do módulo).
        out_path = out_path.with_suffix(".wav")
        out_path.write_bytes(final_audio_path.read_bytes())

    report = {
        "video": str(video_path),
        "output": str(out_path),
        "segments": segment_results,
        "stage_success": build_stage_report(segment_results),
        "assembly": {
            "timeline_discarded_samples": placement.discarded_samples,
            "mix_discarded_samples": mix_discarded_samples,
            "vocals_only": vocals_only,
        },
    }
    (cache_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, nargs="?", default=None)
    parser.add_argument("--r2-key", default=None, help="baixa o vídeo/áudio do bucket R2 por chave")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--vocals-only",
        action="store_true",
        help=(
            "não remixa com o fundo original — só a voz dublada (uso pontual pra "
            "avaliação, não é o comportamento padrão do produto)"
        ),
    )
    args = parser.parse_args()

    if not args.video and not args.r2_key:
        parser.error("informe o caminho do vídeo ou --r2-key")

    if args.r2_key:
        video_name = _sanitize_for_dirname(Path(args.r2_key).name)
        cache_dir = args.cache_dir or Path(".cache/dub") / Path(video_name).stem
        local_video = cache_dir / f"source{Path(args.r2_key).suffix}"
        if args.force or not local_video.exists():
            print(f"baixando {args.r2_key} do R2...")
            r2.download(args.r2_key, local_video)
        video_path = local_video
    else:
        video_path = args.video
        cache_dir = args.cache_dir or Path(".cache/dub") / video_path.stem

    out_path = args.out or video_path.with_name(f"{video_path.stem}.dub{video_path.suffix}")

    report = run_pipeline(video_path, out_path, cache_dir, args.force, vocals_only=args.vocals_only)
    transcript_path, translation_path = save_transcript_and_translation(
        report, cache_dir, Path(report["output"])
    )
    print(json.dumps(report["stage_success"], indent=2, ensure_ascii=False))
    print(f"saída: {report['output']}")
    print(f"relatório completo: {cache_dir / 'report.json'}")
    print(f"transcript: {transcript_path}")
    print(f"tradução usada: {translation_path}")


if __name__ == "__main__":
    main()
