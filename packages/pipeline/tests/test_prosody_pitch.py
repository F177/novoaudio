import librosa
import numpy as np
import pytest

from packages.pipeline.prosody import (
    MAX_PITCH_SHIFT_SEMITONES,
    apply_pitch_contour,
    extract_pitch_contour,
)

SAMPLE_RATE = 16000


def _tone(seconds: float, freq: float = 220.0, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _dominant_freq(audio: np.ndarray) -> float:
    f0 = librosa.yin(audio, fmin=65.0, fmax=1000.0, sr=SAMPLE_RATE)
    return float(np.median(f0))


def test_extract_pitch_contour_flat_tone_is_near_zero() -> None:
    audio = _tone(2.0, freq=220.0)
    contour = extract_pitch_contour(audio, SAMPLE_RATE)
    assert np.all(np.abs(contour) < 0.5)  # tom constante -> pitch relativo ~0 semitons


def test_extract_pitch_contour_rising_pitch_shows_positive_semitones_later() -> None:
    low = _tone(1.0, freq=180.0)
    high = _tone(1.0, freq=260.0)  # bem mais agudo na segunda metade
    audio = np.concatenate([low, high])
    contour = extract_pitch_contour(audio, SAMPLE_RATE)
    half = len(contour) // 2
    assert contour[half:].mean() > contour[:half].mean()


def test_extract_pitch_contour_silent_audio_returns_neutral() -> None:
    silent = np.zeros(int(2.0 * SAMPLE_RATE), dtype=np.float32)
    contour = extract_pitch_contour(silent, SAMPLE_RATE)
    assert np.all(contour == 0.0)


def test_extract_pitch_contour_too_short_returns_neutral() -> None:
    tiny = np.zeros(10, dtype=np.float32)
    contour = extract_pitch_contour(tiny, SAMPLE_RATE)
    assert np.all(contour == 0.0)


def test_apply_pitch_contour_zero_strength_is_noop() -> None:
    audio = _tone(1.0)
    contour = np.array([2.0, -2.0], dtype=np.float32)
    result = apply_pitch_contour(audio, SAMPLE_RATE, contour, strength=0.0)
    assert np.array_equal(result, audio)


def test_apply_pitch_contour_flat_zero_contour_is_effectively_noop() -> None:
    # contorno todo zero -> nenhum bloco desloca de verdade, só passa pelo
    # blend do crossfade (que mistura o bloco com ele mesmo) -- tolerância
    # de ponto flutuante, não igualdade bit-a-bit.
    audio = _tone(1.0)
    contour = np.zeros(5, dtype=np.float32)
    result = apply_pitch_contour(audio, SAMPLE_RATE, contour, strength=1.0)
    assert np.allclose(result, audio, atol=1e-6)


def test_apply_pitch_contour_never_changes_length() -> None:
    audio = _tone(1.3)
    contour = np.array([1.0, -1.0, 2.0], dtype=np.float32)
    result = apply_pitch_contour(audio, SAMPLE_RATE, contour, strength=1.0)
    assert len(result) == len(audio)


def test_apply_pitch_contour_empty_audio_returns_empty() -> None:
    result = apply_pitch_contour(
        np.zeros(0, dtype=np.float32), SAMPLE_RATE, np.array([1.0])
    )
    assert len(result) == 0


def test_apply_pitch_contour_empty_contour_is_noop() -> None:
    audio = _tone(1.0)
    result = apply_pitch_contour(audio, SAMPLE_RATE, np.zeros(0, dtype=np.float32), strength=1.0)
    assert np.array_equal(result, audio)


def test_apply_pitch_contour_actually_shifts_measurable_pitch_up() -> None:
    # 1 janela cobrindo o clipe inteiro (window_seconds >= duração) -> um
    # único pitch_shift de bloco só, mais fácil de medir com precisão.
    audio = _tone(1.0, freq=220.0)
    contour = np.array([2.0], dtype=np.float32)  # +2 semitons constante
    result = apply_pitch_contour(
        audio, SAMPLE_RATE, contour, strength=1.0, window_seconds=1.5
    )
    original_freq = _dominant_freq(audio)
    shifted_freq = _dominant_freq(result)
    expected_freq = original_freq * (2 ** (2.0 / 12.0))
    assert shifted_freq == pytest.approx(expected_freq, rel=0.05)


def test_apply_pitch_contour_shifts_down_for_negative_contour() -> None:
    audio = _tone(1.0, freq=220.0)
    contour = np.array([-3.0], dtype=np.float32)
    result = apply_pitch_contour(
        audio, SAMPLE_RATE, contour, strength=1.0, window_seconds=1.5
    )
    original_freq = _dominant_freq(audio)
    shifted_freq = _dominant_freq(result)
    assert shifted_freq < original_freq


def test_apply_pitch_contour_clamps_extreme_shift() -> None:
    audio = _tone(1.0, freq=220.0)
    extreme_contour = np.array([100.0], dtype=np.float32)  # bem acima do limite
    result = apply_pitch_contour(
        audio, SAMPLE_RATE, extreme_contour, strength=1.0, window_seconds=1.5
    )
    original_freq = _dominant_freq(audio)
    shifted_freq = _dominant_freq(result)
    max_expected_freq = original_freq * (2 ** (MAX_PITCH_SHIFT_SEMITONES / 12.0))
    # tolerância generosa (fase vocoder em deslocamento grande não é exato)
    assert shifted_freq < max_expected_freq * 1.3
