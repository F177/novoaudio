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

from packages.pipeline.synthesis import MAX_TIME_STRETCH_RATIO, MIN_SYNTHESIS_SECONDS

TARGET_LUFS = -14.0
CLIPPING_THRESHOLD = 0.99

# T0.16: acima disso, um descarte é sintoma de bug upstream (segmento mal
# timestampado, vídeo mais curto que o esperado), não arredondamento.
MAX_DISCARDED_SECONDS = 0.5

# Corrigido em 2026-09-18: achado real (beast.mp4) mostrou um caso
# LEGÍTIMO, não-bug, que sozinho excede MAX_DISCARDED_SECONDS — o último
# segmento de um vídeo, curto (abaixo de MIN_SYNTHESIS_SECONDS) e colado no
# fim. Mesmo com o piso do MOSS-TTS comprimido no máximo seguro
# (MAX_TIME_STRETCH_RATIO=1,5x), o mínimo fisicamente alcançável é
# MIN_SYNTHESIS_SECONDS/1.5 ≈ 2,13s — se sobrar menos que isso de vídeo
# depois do início do segmento (comum: a última fala de um vídeo costuma
# ser curta), o descarte É estrutural, sem retry ou compressão que resolva.
#
# A primeira correção (subir MAX_DISCARDED_SECONDS pra 2,5s no agregado)
# foi um erro de design apontado pelo usuário: isso é ajustado a UM vídeo
# de teste, e o produto precisa aguentar qualquer vídeo do público geral.
# Um threshold agregado mais alto esconderia bug real espalhado por VÁRIOS
# segmentos (ex.: 3 segmentos com 0,8s de erro cada, 2,4s no total, cada um
# seria bug isolado mas passaria despercebido). A correção de verdade: o
# caso legítimo só pode afetar UM segmento (o mais próximo do fim do
# vídeo) e só até este teto físico — qualquer descarte além disso, ou
# espalhado por mais de um segmento, continua sendo tratado como bug.
MAX_SINGLE_SEGMENT_DISCARD_SECONDS = (MIN_SYNTHESIS_SECONDS / MAX_TIME_STRETCH_RATIO) + 0.3
MAX_SEGMENTS_WITH_DISCARD = 1


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
    max_single_segment_discard_seconds: float = MAX_SINGLE_SEGMENT_DISCARD_SECONDS,
    max_segments_with_discard: int = MAX_SEGMENTS_WITH_DISCARD,
) -> PlacementResult:
    """Cria uma faixa contínua do tamanho do vídeo original, com cada
    segmento posicionado no timestamp certo e silêncio no resto.

    Achado real e sério, 2026-09-18 (beast.mp4, testando N_CANDIDATES=5): a
    versão anterior desta função só cortava um segmento no FIM do vídeo, e
    assumia que segmentos nunca se sobrepõem entre si porque
    `segmentation.py` garante isso nos dados de ENTRADA. Mas isso ignorava
    que a SÍNTESE pode devolver áudio muito mais longo que o espaço até o
    PRÓXIMO segmento começar — no caso real, um segmento de 10,3s de alvo
    saiu com 52,9s (instabilidade do MOSS-TTS, as 5 amostras independentes
    saíram todas de 14 a 53s, não foi azar de 1 amostra) e teria se
    sobreposto a CINCO segmentos seguintes, corrompendo quase metade do
    vídeo com áudio somado/ininteligível — silenciosamente, sem erro, sem
    flag, porque a verificação só olhava o fim do vídeo, nunca o próximo
    segmento. Agora, cada segmento é cortado no que vier primeiro: o início
    do PRÓXIMO segmento (por `start_seconds`) ou o fim do vídeo. Overlap
    real não acontece mais — vira corte reportado em `discarded_samples`,
    sujeito aos mesmos gates de sanidade abaixo.

    Dois gates independentes decidem se o corte é bug (`TimelineOverflowError`)
    ou o caso legítimo de um segmento curto colado no fim do vídeo/do
    próximo segmento (ver docstring de `MAX_SINGLE_SEGMENT_DISCARD_SECONDS`):

    - Nenhum segmento individual pode descartar mais que
      `max_single_segment_discard_seconds` — acima disso não é mais o piso
      físico do MOSS-TTS, é outra coisa (como o caso real acima).
    - No máximo `max_segments_with_discard` segmentos podem ter QUALQUER
      descarte — o caso legítimo só pode acontecer com um; descarte
      espalhado por vários segmentos é sinal de bug sistemático (timestamps
      errados, `total_duration_seconds` errado), mesmo que cada um sozinho
      pareça pequeno.

    Um vídeo é qualquer coisa que o público geral mandar — estes limiares
    são físicos (derivados de constantes do modelo), não calibrados num
    vídeo de teste específico.
    """
    total_samples = int(round(total_duration_seconds * sample_rate))
    timeline = np.zeros(total_samples, dtype=np.float32)
    per_segment_discard: list[int] = []

    ordered = sorted(segments, key=lambda s: s.start_seconds)
    for i, seg in enumerate(ordered):
        start_sample = int(round(seg.start_seconds * sample_rate))
        if start_sample >= total_samples:
            per_segment_discard.append(len(seg.audio))
            continue

        boundary_sample = (
            int(round(ordered[i + 1].start_seconds * sample_rate))
            if i + 1 < len(ordered)
            else total_samples
        )
        boundary_sample = min(boundary_sample, total_samples)
        room_samples = max(0, boundary_sample - start_sample)

        clip_len = min(len(seg.audio), room_samples)
        per_segment_discard.append(len(seg.audio) - clip_len)
        if clip_len <= 0:
            continue
        end_sample = start_sample + clip_len
        timeline[start_sample:end_sample] += seg.audio[:clip_len]

    discarded_samples = sum(per_segment_discard)
    max_discarded_samples = int(round(max_discarded_seconds * sample_rate))
    max_single_segment_discard_samples = int(
        round(max_single_segment_discard_seconds * sample_rate)
    )
    segments_with_discard = sum(1 for d in per_segment_discard if d > 0)

    if discarded_samples > max_discarded_samples and (
        segments_with_discard > max_segments_with_discard
        or max(per_segment_discard, default=0) > max_single_segment_discard_samples
    ):
        raise TimelineOverflowError(
            f"{discarded_samples} amostras descartadas ao posicionar segmentos na "
            f"timeline ({discarded_samples / sample_rate:.3f}s) em "
            f"{segments_with_discard} segmento(s) — acima do limiar de "
            f"{max_discarded_seconds}s e fora do caso legítimo de 1 segmento no "
            f"piso de duração (até {max_single_segment_discard_seconds:.2f}s), "
            "isso não é arredondamento."
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


def remux_with_video(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    *,
    subtitle_path: Path | None = None,
    subtitle_language: str = "por",
) -> None:
    """Substitui a trilha de áudio do vídeo original pela trilha final, sem
    recodificar vídeo (`-c:v copy`).

    Sem `-shortest` de propósito: a duração de saída tem que bater com o
    vídeo original (critério de aceite), não com o que for mais curto.

    `subtitle_path` (opcional): arquivo `.srt` pra embutir como faixa de
    legenda SELECIONÁVEL (`mov_text`, o codec de legenda padrão pra MP4) —
    o espectador liga/desliga no player, não é "queimada" nos pixels do
    vídeo. `subtitle_language` é a tag ISO 639-2 da faixa (default "por" =
    português).
    """
    if subtitle_path is None:
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
        return

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-i",
            str(subtitle_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-map",
            "2:s:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-c:s",
            "mov_text",
            "-metadata:s:s:0",
            f"language={subtitle_language}",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )
