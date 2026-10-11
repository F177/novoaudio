"""Sinais de voz e emoção para o subagente `dub-judge` (ferramenta de dev).

O juiz não ouve áudio. Este script mede, na CPU e só com o que a execução já
deixou em `.cache/dub/<video>/`, os sinais que ele consegue ler como texto:

- registro: F0 mediano da fonte vs. do dublado e a voz do banco usada
  (`synth_jobs.json` + gênero em `voices.json`). F0 não é gênero: vozes
  masculinas do banco chegam a ~200 Hz;
- locutor: quanto o F0 do dublado varia entre segmentos do mesmo locutor, e
  se dois locutores receberam a mesma voz (T0.31);
- forma da entonação: contorno de F0 em sparkline, fonte e dublado na mesma
  escala;
- pergunta sim/não: em idiomas como o pt-BR ela só se distingue da
  afirmação pela entonação. Pergunta que termina sem pico de F0 pode soar
  como afirmação. A regra é dado do perfil (`evals/languages/<codigo>.json`),
  não código: idioma novo = perfil novo;
- emoção (emotion2vec) e variação de F0 que o `report.json` já tem.

Volume por segmento não é comparado: `match_speech_levels` iguala o nível do
dublado ao da fonte antes da mixagem, então seria tautológico (a mesma
armadilha da T0.20).

Nada aqui é flag de produto nem limiar calibrado. Um sinal só vira flag em
`quality.py` depois de medido contra os rótulos humanos da T0.28.

Uso:
    python -m scripts.judge_audio .cache/dub/<video> [--lang pt-BR]

A escuta humana (original × dublado, com rotulagem) fica em `scripts.listen`.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
VOICE_BANK_MANIFEST = REPO_ROOT / "assets/voices/bank/voices.json"
LANGUAGES_DIR = REPO_ROOT / "evals/languages"

ANALYSIS_SR = 16000
PYIN_FRAME = 2048
PYIN_HOP = 256
HOP_SECONDS = PYIN_HOP / ANALYSIS_SR
FMIN_HZ = 65.0  # mesma faixa de `prosody.extract_pitch_contour`
FMAX_HZ = 1000.0
MIN_VOICED_FRAMES = 5

SPARK_BLOCKS = "▁▂▃▄▅▆▇█"
SPARK_WINDOW_SECONDS = 0.25
SPARK_MAX_CHARS = 40
SPARK_RANGE_ST = 6.0
FINAL_TAIL_SECONDS = 0.8

# Pontos de partida para o juiz decidir o que mandar para escuta.
# Não são limiares de flag; nada aqui foi calibrado contra rótulo humano.
REGISTER_GAP_ST = 5.0
SPEAKER_DRIFT_ST = 3.0
POLAR_PEAK_MIN_ST = 2.0
FLATNESS_RATIO = 0.5  # mesma régua de `quality.PITCH_FLATNESS_RATIO`



@dataclass
class SegmentVoice:
    id: str
    start: float
    end: float
    dub_text: str
    speaker: str | None = None
    voice: str | None = None
    emotion: dict[str, Any] | None = None
    pitch_std_source: float | None = None
    pitch_std_dub: float | None = None
    source_hz: float | None = None
    dub_hz: float | None = None
    source_shape: str = ""
    dub_shape: str = ""
    dub_final_peak_st: float | None = None


# ---------------------------------------------------------------- medidas puras


def f0_track(audio: np.ndarray) -> np.ndarray:
    """F0 em Hz por quadro (áudio a ANALYSIS_SR); NaN onde não há voz."""
    if len(audio) < PYIN_FRAME:
        return np.full(0, np.nan)
    f0, voiced, _ = librosa.pyin(
        audio.astype(np.float32),
        fmin=FMIN_HZ,
        fmax=FMAX_HZ,
        sr=ANALYSIS_SR,
        frame_length=PYIN_FRAME,
        hop_length=PYIN_HOP,
    )
    return np.where(voiced, f0, np.nan)


def median_hz(f0: np.ndarray) -> float | None:
    voiced = f0[~np.isnan(f0)]
    return float(np.median(voiced)) if voiced.size >= MIN_VOICED_FRAMES else None


def semitone_gap(from_hz: float | None, to_hz: float | None) -> float | None:
    if not from_hz or not to_hz:
        return None
    return 12.0 * math.log2(to_hz / from_hz)


def window_contour(f0: np.ndarray, window_seconds: float = SPARK_WINDOW_SECONDS) -> np.ndarray:
    """Semitons relativos à mediana vozeada do próprio trecho, por janela; NaN sem voz."""
    step = max(1, round(window_seconds / HOP_SECONDS))
    n_windows = max(1, math.ceil(len(f0) / step))
    ref = median_hz(f0)
    if ref is None:
        return np.full(n_windows, np.nan)
    st = 12.0 * np.log2(f0 / ref)
    out = np.full(n_windows, np.nan)
    for i in range(n_windows):
        window = st[i * step : (i + 1) * step]
        window = window[~np.isnan(window)]
        if window.size:
            out[i] = window.mean()
    return out


def sparkline(contour: np.ndarray, max_chars: int = SPARK_MAX_CHARS) -> str:
    """Contorno em blocos, escala fixa de ±SPARK_RANGE_ST; `·` = sem voz."""
    if len(contour) > max_chars:
        edges = np.linspace(0, len(contour), max_chars + 1).astype(int)
        merged = []
        for a, b in zip(edges[:-1], edges[1:], strict=True):
            chunk = contour[a:b]
            chunk = chunk[~np.isnan(chunk)]
            merged.append(chunk.mean() if chunk.size else np.nan)
        contour = np.array(merged)
    top = len(SPARK_BLOCKS) - 1
    chars = []
    for v in contour:
        if np.isnan(v):
            chars.append("·")
            continue
        clipped = float(np.clip(v, -SPARK_RANGE_ST, SPARK_RANGE_ST))
        level = (clipped + SPARK_RANGE_ST) / (2 * SPARK_RANGE_ST)
        chars.append(SPARK_BLOCKS[int(round(level * top))])
    return "".join(chars)


def final_peak_st(f0: np.ndarray, tail_seconds: float = FINAL_TAIL_SECONDS) -> float | None:
    """Pico de F0 (p90, robusto a erro de oitava) no fim da fala, em semitons
    acima da mediana do trecho. Pergunta sim/não em pt-BR sobe na última
    sílaba tônica; afirmação tende a cair."""
    ref = median_hz(f0)
    voiced_idx = np.flatnonzero(~np.isnan(f0))
    if ref is None or voiced_idx.size == 0:
        return None
    last = int(voiced_idx[-1])
    tail = f0[max(0, last - round(tail_seconds / HOP_SECONDS)) : last + 1]
    tail = tail[~np.isnan(tail)]
    return float(12.0 * math.log2(np.percentile(tail, 90) / ref))


def load_language(code: str, directory: Path = LANGUAGES_DIR) -> dict[str, Any]:
    path = directory / f"{code}.json"
    if not path.exists():
        raise FileNotFoundError(f"perfil de idioma {path} não existe")
    return json.loads(path.read_text(encoding="utf-8"))


def question_start_regex(profile: dict[str, Any]) -> re.Pattern[str] | None:
    """Regex das palavras interrogativas do perfil, ou None se o idioma não
    marca pergunta sim/não pela entonação (a checagem não se aplica)."""
    rules = profile.get("perguntas") or {}
    if not rules.get("polar_marcada_por_entonacao"):
        return None
    words = sorted(rules.get("inicio_interrogativo", []), key=len, reverse=True)
    alternatives = "|".join(r"\s+".join(map(re.escape, w.split())) for w in words)
    return re.compile(rf"^({alternatives})\b") if alternatives else re.compile(r"^\b\B")


def is_polar_question(text: str, wh_start: re.Pattern[str]) -> bool:
    """A última frase é pergunta sim/não (não começa com palavra interrogativa)."""
    stripped = text.strip()
    if not stripped.endswith("?"):
        return False
    last = re.split(r"(?<=[.!?…])\s+", stripped)[-1]
    last = last.lstrip("¿¡\"'(—–-. ").lower()
    return wh_start.match(last) is None


# ---------------------------------------------------------------- execução


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_voice_genders(manifest: Path = VOICE_BANK_MANIFEST) -> dict[str, str]:
    return {v["id"]: v.get("gender", "?") for v in _read_json(manifest) or []}


def load_segments(cache_dir: Path) -> list[SegmentVoice]:
    """Dados baratos (sem áudio): tempos, texto, locutor, voz e emoção."""
    report = _read_json(cache_dir / "report.json")
    if report is None:
        raise FileNotFoundError(f"{cache_dir / 'report.json'} não existe; rode scripts/dub.py")
    speakers = {s["id"]: s.get("speaker") for s in _read_json(cache_dir / "segments.json") or []}
    voices = {
        j["id"]: Path(j["reference_audio_path"]).stem
        for j in _read_json(cache_dir / "synth_jobs.json") or []
        if j.get("reference_audio_path")
    }
    return [
        SegmentVoice(
            id=s["id"],
            start=float(s["t_inicio"]),
            end=float(s["t_fim"]),
            dub_text=s.get("traducao") or "",
            speaker=speakers.get(s["id"]),
            voice=voices.get(s["id"]),
            emotion=s.get("emotion_match"),
            pitch_std_source=s.get("pitch_std_source"),
            pitch_std_dub=s.get("pitch_std_dub"),
        )
        for s in report["segments"]
    ]


def measure_pitch(cache_dir: Path, segments: list[SegmentVoice]) -> None:
    """Preenche F0 e contornos (librosa.pyin na CPU, ~3 s por 10 s de áudio)."""
    source_path = cache_dir / "vocals.wav"
    if not source_path.exists():
        source_path = cache_dir / "audio.wav"
    for seg in segments:
        if source_path.exists():
            src, _ = librosa.load(
                source_path, sr=ANALYSIS_SR, offset=seg.start, duration=seg.end - seg.start
            )
            f0 = f0_track(src)
            seg.source_hz, seg.source_shape = median_hz(f0), sparkline(window_contour(f0))
        dub_path = cache_dir / "final_segments" / f"{seg.id}.wav"
        if dub_path.exists():
            dub, _ = librosa.load(dub_path, sr=ANALYSIS_SR)
            f0 = f0_track(dub)
            seg.dub_hz, seg.dub_shape = median_hz(f0), sparkline(window_contour(f0))
            seg.dub_final_peak_st = final_peak_st(f0)


# ---------------------------------------------------------------- saída


def _hz(v: float | None) -> str:
    return f"{v:.0f} Hz" if v else "—"


def _median(values: list[float | None]) -> float | None:
    present = [v for v in values if v]
    return float(np.median(present)) if present else None


def speaker_summary(segments: list[SegmentVoice], genders: dict[str, str]) -> list[str]:
    by_speaker: dict[str, list[SegmentVoice]] = defaultdict(list)
    for s in segments:
        by_speaker[s.speaker or "?"].append(s)
    speakers_by_voice: dict[str, set[str]] = defaultdict(set)
    lines = []
    for speaker, segs in sorted(by_speaker.items()):
        voices = sorted({s.voice for s in segs if s.voice})
        for v in voices:
            speakers_by_voice[v].add(speaker)
        voice_txt = ", ".join(f"{v} ({genders.get(v, '?')})" for v in voices) or "?"
        src, dub = _median([s.source_hz for s in segs]), _median([s.dub_hz for s in segs])
        line = f"{speaker} ({len(segs)} seg): voz {voice_txt} · fonte {_hz(src)} · dub {_hz(dub)}"
        gap = semitone_gap(src, dub)
        if gap is not None:
            line += f" ({gap:+.1f} st)" + ("  ⚠ registro" if abs(gap) > REGISTER_GAP_ST else "")
        drifts = [abs(semitone_gap(dub, s.dub_hz) or 0.0) for s in segs if s.dub_hz]
        if drifts:
            drift = max(drifts)
            line += f" · desvio máx. entre segmentos {drift:.1f} st"
            line += "  ⚠ instável" if drift > SPEAKER_DRIFT_ST else ""
        lines.append(line)
    for voice, speakers in sorted(speakers_by_voice.items()):
        if len(speakers) > 1:
            lines.append(f"⚠ a voz {voice} foi usada para {', '.join(sorted(speakers))}")
    return lines


def render(
    name: str,
    segments: list[SegmentVoice],
    genders: dict[str, str],
    wh_start: re.Pattern[str] | None = None,
) -> str:
    lines = [f"# {name} — sinais de voz e emoção (medidos, não ouvidos)", "", "## Por locutor"]
    lines += speaker_summary(segments, genders)
    lines += ["", "## Por segmento"]
    for s in segments:
        head = f"{s.id} {s.start:.2f}–{s.end:.2f}"
        head += (f" {s.speaker}" if s.speaker else "") + (f" voz {s.voice}" if s.voice else "")
        lines.append(head)
        reg = f"  F0: fonte {_hz(s.source_hz)} · dub {_hz(s.dub_hz)}"
        gap = semitone_gap(s.source_hz, s.dub_hz)
        if gap is not None:
            reg += f" ({gap:+.1f} st)" + ("  ⚠ registro" if abs(gap) > REGISTER_GAP_ST else "")
        lines.append(reg)
        if s.source_shape or s.dub_shape:
            lines.append(f"  forma: fonte {s.source_shape or '—'}")
            lines.append(f"         dub   {s.dub_shape or '—'}")
        if s.pitch_std_source and s.pitch_std_dub is not None:
            ratio = s.pitch_std_dub / s.pitch_std_source
            flat = "  ⚠ achatado" if ratio < FLATNESS_RATIO else ""
            lines.append(
                f"  variação F0: fonte {s.pitch_std_source:.1f} st · "
                f"dub {s.pitch_std_dub:.1f} st ({ratio:.2f}x){flat}"
            )
        if s.emotion:
            e = s.emotion
            lines.append(
                f"  emoção: fonte {e.get('source_emotion')} {e.get('source_emotion_score', 0):.2f}"
                f" · dub {e.get('dub_emotion')} {e.get('dub_emotion_score', 0):.2f}"
                f" (sim {e.get('similarity', 0):.2f})"
            )
        if wh_start is not None and is_polar_question(s.dub_text, wh_start):
            peak = s.dub_final_peak_st
            note = "—" if peak is None else f"{peak:+.1f} st"
            if peak is not None and peak < POLAR_PEAK_MIN_ST:
                note += "  ⚠ pode soar como afirmação"
            lines.append(f"  pergunta sim/não no fim: pico final do dub {note}")
    if wh_start is None:
        lines += ["", "(checagem de pergunta desligada: o perfil do idioma não a define)"]
    if not any(s.emotion for s in segments):
        lines += ["", "(emoção não medida nesta execução: rode dub.py com --emotion-match)"]
    return "\n".join(lines)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # console cp1252 do Windows quebra nos símbolos
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("cache_dir", type=Path, help=".cache/dub/<video>")
    parser.add_argument("--lang", default="pt-BR", help="perfil em evals/languages/")
    args = parser.parse_args()

    segments = load_segments(args.cache_dir)
    measure_pitch(args.cache_dir, segments)
    wh_start = question_start_regex(load_language(args.lang))
    print(render(args.cache_dir.name, segments, load_voice_genders(), wh_start))


if __name__ == "__main__":
    main()
