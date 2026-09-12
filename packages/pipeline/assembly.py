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


@dataclass
class TimedAudio:
    start_seconds: float
    audio: np.ndarray  # mono, float32, amplitude em [-1, 1]


def place_segments_on_timeline(
    segments: list[TimedAudio], total_duration_seconds: float, sample_rate: int
) -> np.ndarray:
    """Cria uma faixa contínua do tamanho do vídeo original, com cada
    segmento posicionado no timestamp certo e silêncio no resto.

    Segmentos que se sobrepõem (não deveriam — `segmentation.py` garante
    isso) são somados, não é tratado como erro aqui: essa invariante é de
    quem gera os segmentos, não desta função.
    """
    total_samples = int(round(total_duration_seconds * sample_rate))
    timeline = np.zeros(total_samples, dtype=np.float32)

    for seg in segments:
        start_sample = int(round(seg.start_seconds * sample_rate))
        if start_sample >= total_samples:
            continue
        end_sample = min(start_sample + len(seg.audio), total_samples)
        clip_len = end_sample - start_sample
        if clip_len <= 0:
            continue
        timeline[start_sample:end_sample] += seg.audio[:clip_len]

    return timeline


def mix_with_background(
    vocals: np.ndarray, background: np.ndarray, background_gain: float = 1.0
) -> np.ndarray:
    """Mixa a trilha de vocais dublados com o stem de fundo (Demucs).

    Se os tamanhos diferirem (ex.: arredondamento de alguns samples), corta
    no mais curto — não deveria acontecer se os dois vierem do mesmo
    vídeo, mas não trava o pipeline por causa de 1-2 amostras de diferença.
    """
    n = min(len(vocals), len(background))
    return vocals[:n] + background[:n] * background_gain


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
