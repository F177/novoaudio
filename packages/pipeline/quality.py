"""Gates de qualidade.

Emite flags automáticas de suspeita de erro por segmento, pra alimentar o
editor (T2.x, ver CLAUDE.md glossário "Flag"): cada flag marca um jeito
específico de o pipeline ter "chutado" nesse segmento — não é veredito
final, o humano decide no editor.

Duas categorias, dependendo de onde mora o cálculo:

- **Calculável aqui** (`fora_duracao`, `clipping`, `silencio_anormal`): a
  métrica bruta é pura (duração/amplitude/RMS de um array numpy), então a
  função inteira mora neste módulo.
- **Calculada em GPU, só o limiar mora aqui** (`cer_alto`,
  `traducao_infiel`, `locutor_suspeito`): CER (`runpod/evaluate`),
  similaridade fonte/tradução (embeddings) e similaridade de locutor
  (WavLM-TDNN/ECAPA) exigem modelo — este módulo recebe o score já
  calculado por quem chama e só aplica o limiar. Não reimplementa nem
  chama modelo (regra de dependência de `packages/pipeline`, CLAUDE.md).

Limiares default são ponto de partida (não medidos em produção ainda) —
ajustar com dado real do editor (quantos segmentos com a flag o usuário
de fato corrige) assim que houver volume.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from packages.pipeline.budget import tolerance_range

CER_THRESHOLD = 0.15
TRANSLATION_FIDELITY_THRESHOLD = 0.75
SPEAKER_SIMILARITY_THRESHOLD = 0.75
CLIPPING_THRESHOLD = 0.99

# RMS abaixo disso conta como "silencioso" numa janela; achado do T0.6 é que
# o chão de ruído de pausas naturais fica em torno de -45dBFS (~0.0056 linear)
# — usar um limiar um pouco acima disso (~-40dBFS) evita marcar silêncio real
# de pausa natural como se fosse janela de fala.
SILENCE_RMS_THRESHOLD = 0.01
SILENCE_WINDOW_SECONDS = 0.05
SILENCE_MARGIN_SECONDS = 0.5


def fora_duracao(achieved_seconds: float, target_seconds: float, *, on_screen: bool) -> bool:
    """Duração real fora da tolerância de isocronia (±8% on-screen, ±20% off-screen)."""
    low, high = tolerance_range(target_seconds, on_screen=on_screen)
    return not (low <= achieved_seconds <= high)


def cer_alto(cer: float, threshold: float = CER_THRESHOLD) -> bool:
    """CER round-trip (CLAUDE.md glossário) acima do limiar aceitável."""
    return cer > threshold


def traducao_infiel(
    fidelity_score: float, threshold: float = TRANSLATION_FIDELITY_THRESHOLD
) -> bool:
    """Score de fidelidade (similaridade semântica fonte/tradução) abaixo do limiar.

    O score em si (embeddings ou LLM-juiz) é calculado fora deste módulo —
    ver docstring de `translation.py` sobre por que fidelidade factual não é
    algo que uma função pura consiga avaliar sozinha.
    """
    return fidelity_score < threshold


def clipping(audio: np.ndarray, threshold: float = CLIPPING_THRESHOLD) -> bool:
    return bool(np.any(np.abs(audio) > threshold))


def silencio_anormal(
    audio: np.ndarray,
    sample_rate: int,
    expected_pause_seconds: float,
    *,
    rms_threshold: float = SILENCE_RMS_THRESHOLD,
    margin_seconds: float = SILENCE_MARGIN_SECONDS,
    window_seconds: float = SILENCE_WINDOW_SECONDS,
) -> bool:
    """Sinaliza quando o segmento tem mais silêncio do que o esperado.

    `expected_pause_seconds` é a soma das pausas internas já conhecidas
    (`segmentation.InternalPause` + qualquer `[pause X.Ys]` injetada,
    ver `synthesis.py`). Mede silêncio real em janelas de RMS e compara
    contra esse orçamento mais uma margem — silêncio muito acima disso
    normalmente indica falha de síntese (ex.: o TTS "travou" e devolveu
    áudio maior com trecho morto), não pausa intencional.
    """
    window_samples = max(1, int(window_seconds * sample_rate))
    n_windows = len(audio) // window_samples
    if n_windows == 0:
        return False

    trimmed = audio[: n_windows * window_samples].reshape(n_windows, window_samples)
    rms = np.sqrt(np.mean(trimmed.astype(np.float64) ** 2, axis=1))
    silent_seconds = float(np.sum(rms < rms_threshold)) * window_seconds

    return silent_seconds > expected_pause_seconds + margin_seconds


def locutor_suspeito(
    speaker_similarity: float, threshold: float = SPEAKER_SIMILARITY_THRESHOLD
) -> bool:
    """Similaridade de locutor (WavLM-TDNN/ECAPA, CLAUDE.md) abaixo do limiar.

    Sinaliza quando a voz clonada usada na síntese não soa como o "perfil de
    voz" esperado do canal/locutor — o score em si vem de fora.
    """
    return speaker_similarity < threshold


@dataclass
class QualityFlags:
    fora_duracao: bool
    cer_alto: bool
    traducao_infiel: bool
    clipping: bool
    silencio_anormal: bool
    locutor_suspeito: bool

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def evaluate_segment(
    *,
    achieved_seconds: float,
    target_seconds: float,
    on_screen: bool,
    cer: float,
    fidelity_score: float,
    audio: np.ndarray,
    sample_rate: int,
    expected_pause_seconds: float,
    speaker_similarity: float,
) -> QualityFlags:
    """Roda todos os gates pra um segmento e monta o conjunto de flags."""
    return QualityFlags(
        fora_duracao=fora_duracao(achieved_seconds, target_seconds, on_screen=on_screen),
        cer_alto=cer_alto(cer),
        traducao_infiel=traducao_infiel(fidelity_score),
        clipping=clipping(audio),
        silencio_anormal=silencio_anormal(audio, sample_rate, expected_pause_seconds),
        locutor_suspeito=locutor_suspeito(speaker_similarity),
    )
