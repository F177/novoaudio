"""Montagem e mix.

Posiciona cada segmento sintetizado na timeline original, mixa com o stem
de fundo (Demucs), normaliza loudness (EBU R128, alvo −14 LUFS) e remuxa
o resultado com o vídeo original via ffmpeg.

A parte de timeline/mix/normalização é pura (numpy + `pyloudnorm`, sem
chamada externa) — só o remux final precisa mesmo do ffmpeg via
subprocess, porque combinar áudio com um stream de vídeo não é algo que
valha a pena reimplementar. `pyloudnorm` é uma implementação em Python
puro do ITU-R BS.1770/EBU R128, escolhida no lugar do filtro `loudnorm`
do ffmpeg justamente pra manter essa etapa testável com dado sintético,
sem precisar de subprocess.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyloudnorm as pyln

TARGET_LUFS = -14.0
CLIPPING_THRESHOLD = 0.99

# T0.16: quanto pode ser descartado silenciosamente antes de considerarmos
# bug em vez de arredondamento. Meio segundo de áudio perdido é sempre um
# sintoma de segmento mal timestampado ou vídeo mais curto que o esperado —
# arredondamento de sample-rate nunca chega perto disso.
MAX_DISCARDED_SECONDS = 0.5


class TimelineOverflowError(RuntimeError):
    """Corte de áudio maior que `MAX_DISCARDED_SECONDS` — não é arredondamento."""


@dataclass
class TimedAudio:
    start_seconds: float
    audio: np.ndarray  # mono, float32, amplitude em [-1, 1]


@dataclass
class PlacementResult:
    timeline: np.ndarray
    discarded_samples: int


@dataclass
class MixResult:
    audio: np.ndarray
    discarded_samples: int


def place_segments_on_timeline(
    segments: list[TimedAudio],
    total_duration_seconds: float,
    sample_rate: int,
    *,
    max_discarded_seconds: float = MAX_DISCARDED_SECONDS,
) -> PlacementResult:
    """Cria uma faixa contínua do tamanho do vídeo original, com cada
    segmento posicionado no timestamp certo e silêncio no resto.

    Segmentos que se sobrepõem (não deveriam — `segmentation.py` garante
    isso) são somados, não é tratado como erro aqui: essa invariante é de
    quem gera os segmentos, não desta função.

    Segmento que começa depois do fim do vídeo, ou cuja síntese estourou o
    fim da timeline, tem a parte que não coube descartada — isso é
    reportado em `discarded_samples`, não engolido em silêncio. Acima de
    `max_discarded_seconds` no total, levanta `TimelineOverflowError`: a
    essa altura não é mais arredondamento de sample, é sintoma de bug
    upstream (segmento mal timestampado, síntese que ignorou o orçamento).
    """
    total_samples = int(round(total_duration_seconds * sample_rate))
    timeline = np.zeros(total_samples, dtype=np.float32)
    discarded_samples = 0

    for seg in segments:
        start_sample = int(round(seg.start_seconds * sample_rate))
        if start_sample >= total_samples:
            discarded_samples += len(seg.audio)
            continue
        end_sample = min(start_sample + len(seg.audio), total_samples)
        clip_len = end_sample - start_sample
        discarded_samples += len(seg.audio) - clip_len
        if clip_len <= 0:
            continue
        timeline[start_sample:end_sample] += seg.audio[:clip_len]

    max_discarded_samples = int(round(max_discarded_seconds * sample_rate))
    if discarded_samples > max_discarded_samples:
        raise TimelineOverflowError(
            f"{discarded_samples} amostras descartadas ao posicionar segmentos na "
            f"timeline ({discarded_samples / sample_rate:.3f}s) — acima do limiar "
            f"de {max_discarded_seconds}s configurado, isso não é arredondamento."
        )

    return PlacementResult(timeline=timeline, discarded_samples=discarded_samples)


def mix_with_background(
    vocals: np.ndarray,
    background: np.ndarray,
    sample_rate: int,
    background_gain: float = 1.0,
    *,
    max_discarded_seconds: float = MAX_DISCARDED_SECONDS,
) -> MixResult:
    """Mixa a trilha de vocais dublados com o stem de fundo (Demucs).

    Se os tamanhos diferirem (ex.: arredondamento de alguns samples), corta
    no mais curto — não deveria acontecer se os dois vierem do mesmo
    vídeo, mas não trava o pipeline por causa de 1-2 amostras de diferença.
    A diferença descartada é reportada em `discarded_samples`; acima de
    `max_discarded_seconds` levanta `TimelineOverflowError` (T0.16) — uma
    divergência grande entre vocais e fundo é sinal de que vieram de fontes
    diferentes, não arredondamento.
    """
    n = min(len(vocals), len(background))
    discarded_samples = abs(len(vocals) - len(background))

    max_discarded_samples = int(round(max_discarded_seconds * sample_rate))
    if discarded_samples > max_discarded_samples:
        raise TimelineOverflowError(
            f"{discarded_samples} amostras de diferença entre vocais e fundo "
            f"({discarded_samples / sample_rate:.3f}s) — acima do limiar de "
            f"{max_discarded_seconds}s configurado, isso não é arredondamento."
        )

    mixed = vocals[:n] + background[:n] * background_gain
    return MixResult(audio=mixed, discarded_samples=discarded_samples)


def has_clipping(audio: np.ndarray, threshold: float = CLIPPING_THRESHOLD) -> bool:
    return bool(np.any(np.abs(audio) > threshold))


def normalize_loudness(
    audio: np.ndarray, sample_rate: int, target_lufs: float = TARGET_LUFS
) -> np.ndarray:
    """Normaliza loudness integrado pro alvo EBU R128 (−14 LUFS por padrão)."""
    meter = pyln.Meter(sample_rate)
    current_loudness = meter.integrated_loudness(audio)
    if current_loudness == float("-inf"):
        return audio  # áudio totalmente silencioso, não há o que normalizar
    return pyln.normalize.loudness(audio, current_loudness, target_lufs)


def remux_with_video(video_path: Path, audio_path: Path, output_path: Path) -> None:
    """Substitui a trilha de áudio do vídeo original pela trilha final, sem
    recodificar vídeo (`-c:v copy`).

    Sem `-shortest` de propósito: a duração de saída tem que bater com o
    vídeo original (critério de aceite), não com o que for mais curto.
    """
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )
