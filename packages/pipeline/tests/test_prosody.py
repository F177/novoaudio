import numpy as np
import pytest

from packages.pipeline.prosody import (
    apply_energy_envelope,
    extract_energy_envelope,
    resample_envelope,
)

SAMPLE_RATE = 16000


def _tone(seconds: float, amplitude: float = 0.5, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_extract_energy_envelope_normalizes_to_peak() -> None:
    quiet = _tone(0.5, amplitude=0.1)
    loud = _tone(0.5, amplitude=0.5)
    audio = np.concatenate([quiet, loud])
    envelope = extract_energy_envelope(audio, SAMPLE_RATE)
    assert envelope.max() == pytest.approx(1.0, abs=1e-3)
    # metade inicial (quieta) deve ter energia relativa bem menor que a final
    half = len(envelope) // 2
    assert envelope[:half].mean() < envelope[half:].mean()


def test_extract_energy_envelope_empty_audio_returns_neutral() -> None:
    envelope = extract_energy_envelope(np.zeros(0, dtype=np.float32), SAMPLE_RATE)
    assert np.all(envelope == 1.0)


def test_extract_energy_envelope_silent_audio_returns_neutral() -> None:
    silent = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)
    envelope = extract_energy_envelope(silent, SAMPLE_RATE)
    assert np.all(envelope == 1.0)


def test_resample_envelope_stretches_to_target_length() -> None:
    envelope = np.array([0.0, 1.0], dtype=np.float32)
    result = resample_envelope(envelope, target_length=10)
    assert len(result) == 10
    assert result[0] == pytest.approx(0.0)
    assert result[-1] == pytest.approx(1.0)


def test_resample_envelope_same_length_is_noop() -> None:
    envelope = np.array([0.2, 0.8, 0.5], dtype=np.float32)
    result = resample_envelope(envelope, target_length=3)
    assert np.array_equal(result, envelope)


def test_resample_envelope_empty_source_returns_neutral() -> None:
    result = resample_envelope(np.zeros(0, dtype=np.float32), target_length=5)
    assert np.all(result == 1.0)
    assert len(result) == 5


def test_apply_energy_envelope_zero_strength_is_noop() -> None:
    audio = _tone(1.0, amplitude=0.5)
    envelope = np.array([0.1, 1.0], dtype=np.float32)
    result = apply_energy_envelope(audio, SAMPLE_RATE, envelope, strength=0.0)
    assert np.array_equal(result, audio)


def test_apply_energy_envelope_never_changes_length() -> None:
    audio = _tone(1.3, amplitude=0.5)
    envelope = np.array([0.2, 0.9, 0.4], dtype=np.float32)
    result = apply_energy_envelope(audio, SAMPLE_RATE, envelope, strength=0.7)
    assert len(result) == len(audio)


def test_apply_energy_envelope_amplifies_where_source_was_louder() -> None:
    # sintetizado com volume constante; origem tinha um pico bem no meio —
    # depois de aplicar, o meio do áudio sintetizado deve ficar mais alto
    # que o início/fim.
    audio = _tone(3.0, amplitude=0.3)
    envelope = np.array([0.1, 0.1, 1.0, 0.1, 0.1], dtype=np.float32)
    result = apply_energy_envelope(audio, SAMPLE_RATE, envelope, strength=1.0)

    window = len(audio) // 5
    start_energy = np.abs(result[:window]).mean()
    middle_energy = np.abs(result[2 * window : 3 * window]).mean()
    assert middle_energy > start_energy


def test_apply_energy_envelope_flat_source_barely_changes_audio() -> None:
    audio = _tone(1.0, amplitude=0.5)
    flat_envelope = np.ones(10, dtype=np.float32)
    result = apply_energy_envelope(audio, SAMPLE_RATE, flat_envelope, strength=1.0)
    assert np.allclose(result, audio, atol=1e-5)


def test_apply_energy_envelope_clamps_gain_to_safe_range() -> None:
    # envelope com contraste extremo não deve conseguir apagar o áudio nem
    # estourar amplitude muito além do original. Usa um sinal constante
    # (não senoidal) pra evitar falso positivo nas cruzadas por zero do
    # seno, onde a razão saída/entrada é instável (0/quase-zero).
    audio = np.full(SAMPLE_RATE, 0.4, dtype=np.float32)
    extreme_envelope = np.array([0.0, 5.0], dtype=np.float32)
    result = apply_energy_envelope(audio, SAMPLE_RATE, extreme_envelope, strength=1.0)
    ratio = np.abs(result) / np.abs(audio)
    assert ratio.max() <= 2.0 + 1e-6
    assert ratio.min() >= 0.3 - 1e-6


def test_apply_energy_envelope_empty_audio_returns_empty() -> None:
    result = apply_energy_envelope(
        np.zeros(0, dtype=np.float32), SAMPLE_RATE, np.array([1.0])
    )
    assert len(result) == 0
