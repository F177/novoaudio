import numpy as np
import pytest

from packages.pipeline.prosody import apply_energy_envelope, extract_pitch_contour
from packages.pipeline.quality import (
    EMOTION_SIMILARITY_THRESHOLD,
    PITCH_FLATNESS_RATIO,
    emocao_incompativel,
    emotion_similarity,
    entonacao_desalinhada,
    extract_pitch_stats,
    prosody_correlation,
)


def test_prosody_correlation_identical_contours_is_one() -> None:
    contour = np.array([0.1, 0.5, 0.9, 0.3, 0.2], dtype=np.float32)
    assert prosody_correlation(contour, contour.copy()) == pytest.approx(1.0, abs=1e-6)


def test_prosody_correlation_inverted_contour_is_negative_one() -> None:
    contour = np.array([0.1, 0.5, 0.9, 0.3, 0.2], dtype=np.float32)
    inverted = -contour
    assert prosody_correlation(contour, inverted) == pytest.approx(-1.0, abs=1e-6)


def test_prosody_correlation_unrelated_contours_is_low() -> None:
    # um sobe monotonicamente, o outro é constante com um ruído simétrico —
    # sem relação linear real
    rising = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    unrelated = np.array([0.5, 0.1, 0.9, 0.2, 0.5], dtype=np.float32)
    correlation = prosody_correlation(rising, unrelated)
    assert -0.5 < correlation < 0.5


def test_prosody_correlation_flat_source_returns_zero_not_nan() -> None:
    flat = np.ones(5, dtype=np.float32)
    varying = np.array([0.1, 0.5, 0.9, 0.3, 0.2], dtype=np.float32)
    result = prosody_correlation(flat, varying)
    assert result == 0.0


def test_prosody_correlation_flat_dub_returns_zero_not_nan() -> None:
    varying = np.array([0.1, 0.5, 0.9, 0.3, 0.2], dtype=np.float32)
    flat = np.zeros(5, dtype=np.float32)
    result = prosody_correlation(varying, flat)
    assert result == 0.0


def test_prosody_correlation_too_short_returns_zero() -> None:
    assert prosody_correlation(np.array([1.0]), np.array([1.0])) == 0.0


def test_prosody_correlation_raises_on_length_mismatch() -> None:
    with pytest.raises(ValueError):
        prosody_correlation(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


SAMPLE_RATE = 16000


def _rising_tone(seconds: float = 2.0) -> np.ndarray:
    # concatena dois tons de frequência diferente — gera variação real de
    # pitch pro `extract_pitch_contour` detectar (um tom constante teria
    # desvio ~0, não serve pra testar as estatísticas agregadas)
    half = seconds / 2
    t = np.arange(int(half * SAMPLE_RATE)) / SAMPLE_RATE
    low = (0.5 * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)
    high = (0.5 * np.sin(2 * np.pi * 260.0 * t)).astype(np.float32)
    return np.concatenate([low, high])


def test_extract_pitch_stats_varying_pitch_has_nonzero_range_and_std() -> None:
    contour = extract_pitch_contour(_rising_tone(), SAMPLE_RATE)
    stats = extract_pitch_stats(contour)
    assert stats["range"] > 0.0
    assert stats["std"] > 0.0


def test_extract_pitch_stats_all_unvoiced_is_zero() -> None:
    silent_contour = np.zeros(10, dtype=np.float32)
    stats = extract_pitch_stats(silent_contour)
    assert stats == {"range": 0.0, "std": 0.0, "slope": 0.0}


def test_extract_pitch_stats_single_voiced_frame_is_zero() -> None:
    contour = np.array([0.0, 0.0, 2.5, 0.0], dtype=np.float32)
    stats = extract_pitch_stats(contour)
    assert stats == {"range": 0.0, "std": 0.0, "slope": 0.0}


def test_extract_pitch_stats_rising_contour_has_positive_slope() -> None:
    contour = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    stats = extract_pitch_stats(contour)
    assert stats["slope"] > 0.0


def test_extract_pitch_stats_not_altered_by_energy_envelope() -> None:
    # T0.20, critério de aceite: a transferência de envelope de energia
    # (`apply_energy_envelope`, só muda ganho/amplitude) não pode alterar
    # a nova métrica de entonação — diferente da correlação de energia
    # antiga, que era calculada EM CIMA do próprio contorno que a
    # transferência impunha (tautológica).
    audio = _rising_tone()
    stats_before = extract_pitch_stats(extract_pitch_contour(audio, SAMPLE_RATE))

    strong_envelope = np.array([0.2, 1.0, 1.8, 0.4], dtype=np.float32)
    modulated = apply_energy_envelope(audio, SAMPLE_RATE, strong_envelope, strength=1.0)
    stats_after = extract_pitch_stats(extract_pitch_contour(modulated, SAMPLE_RATE))

    assert stats_after["range"] == pytest.approx(stats_before["range"], abs=0.1)
    assert stats_after["std"] == pytest.approx(stats_before["std"], abs=0.1)
    assert stats_after["slope"] == pytest.approx(stats_before["slope"], abs=0.1)


def test_entonacao_desalinhada_dub_much_flatter_than_source_is_true() -> None:
    assert entonacao_desalinhada(source_pitch_std=2.0, dub_pitch_std=0.5) is True


def test_entonacao_desalinhada_dub_similar_variation_is_false() -> None:
    assert entonacao_desalinhada(source_pitch_std=2.0, dub_pitch_std=1.8) is False


def test_entonacao_desalinhada_zero_source_std_never_flags() -> None:
    # fonte sem variação de pitch detectável não tem base de comparação —
    # ausência de sinal na FONTE não é evidência de desalinhamento
    assert entonacao_desalinhada(source_pitch_std=0.0, dub_pitch_std=0.0) is False
    assert entonacao_desalinhada(source_pitch_std=0.0, dub_pitch_std=5.0) is False


def test_entonacao_desalinhada_custom_ratio() -> None:
    assert entonacao_desalinhada(source_pitch_std=2.0, dub_pitch_std=1.5, min_ratio=0.9) is True
    assert entonacao_desalinhada(source_pitch_std=2.0, dub_pitch_std=1.9, min_ratio=0.5) is False


def test_entonacao_desalinhada_default_ratio_matches_constant() -> None:
    source_std = 2.0
    boundary = source_std * PITCH_FLATNESS_RATIO
    assert entonacao_desalinhada(source_std, boundary - 0.01) is True
    assert entonacao_desalinhada(source_std, boundary + 0.01) is False


def test_emotion_similarity_identical_scores_is_one() -> None:
    scores = np.array([0.7, 0.1, 0.1, 0.1], dtype=np.float32)
    assert emotion_similarity(scores, scores.copy()) == pytest.approx(1.0, abs=1e-6)


def test_emotion_similarity_orthogonal_scores_is_zero() -> None:
    source = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    dub = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    assert emotion_similarity(source, dub) == pytest.approx(0.0, abs=1e-6)


def test_emotion_similarity_opposite_scores_is_negative_one() -> None:
    source = np.array([1.0, 0.0], dtype=np.float32)
    dub = np.array([-1.0, 0.0], dtype=np.float32)
    assert emotion_similarity(source, dub) == pytest.approx(-1.0, abs=1e-6)


def test_emotion_similarity_zero_vector_returns_zero_not_nan() -> None:
    zeros = np.zeros(4, dtype=np.float32)
    scores = np.array([0.7, 0.1, 0.1, 0.1], dtype=np.float32)
    assert emotion_similarity(zeros, scores) == 0.0
    assert emotion_similarity(scores, zeros) == 0.0


def test_emotion_similarity_raises_on_length_mismatch() -> None:
    with pytest.raises(ValueError):
        emotion_similarity(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


def test_emocao_incompativel_below_threshold() -> None:
    assert emocao_incompativel(0.1) is True
    assert emocao_incompativel(EMOTION_SIMILARITY_THRESHOLD - 0.01) is True


def test_emocao_incompativel_above_threshold() -> None:
    assert emocao_incompativel(0.9) is False
    assert emocao_incompativel(EMOTION_SIMILARITY_THRESHOLD + 0.01) is False


def test_emocao_incompativel_custom_threshold() -> None:
    assert emocao_incompativel(0.5, threshold=0.6) is True
    assert emocao_incompativel(0.7, threshold=0.6) is False
