import numpy as np
import pytest

from packages.pipeline.quality import (
    EMOTION_SIMILARITY_THRESHOLD,
    PROSODY_CORRELATION_THRESHOLD,
    emocao_incompativel,
    emotion_similarity,
    entonacao_desalinhada,
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


def test_entonacao_desalinhada_below_threshold() -> None:
    assert entonacao_desalinhada(0.1) is True
    assert entonacao_desalinhada(PROSODY_CORRELATION_THRESHOLD - 0.01) is True


def test_entonacao_desalinhada_above_threshold() -> None:
    assert entonacao_desalinhada(0.8) is False
    assert entonacao_desalinhada(PROSODY_CORRELATION_THRESHOLD + 0.01) is False


def test_entonacao_desalinhada_custom_threshold() -> None:
    assert entonacao_desalinhada(0.5, threshold=0.6) is True
    assert entonacao_desalinhada(0.7, threshold=0.6) is False


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
