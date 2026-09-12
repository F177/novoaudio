import json

import numpy as np

from packages.pipeline.quality import (
    QualityFlags,
    cer_alto,
    clipping,
    evaluate_segment,
    fora_duracao,
    locutor_suspeito,
    silencio_anormal,
    traducao_infiel,
)

SAMPLE_RATE = 16000


def _tone(seconds: float, amplitude: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_fora_duracao_true_when_outside_tolerance() -> None:
    assert fora_duracao(achieved_seconds=15.0, target_seconds=10.0, on_screen=True) is True


def test_fora_duracao_false_when_within_tolerance() -> None:
    assert fora_duracao(achieved_seconds=10.5, target_seconds=10.0, on_screen=True) is False


def test_cer_alto_true_above_threshold() -> None:
    assert cer_alto(0.30) is True


def test_cer_alto_false_below_threshold() -> None:
    assert cer_alto(0.05) is False


def test_traducao_infiel_true_below_threshold() -> None:
    assert traducao_infiel(0.40) is True


def test_traducao_infiel_false_above_threshold() -> None:
    assert traducao_infiel(0.95) is False


def test_clipping_true_when_over_threshold() -> None:
    audio = np.array([0.1, 1.5, -0.2], dtype=np.float32)
    assert clipping(audio) is True


def test_clipping_false_for_normal_audio() -> None:
    assert clipping(_tone(1.0)) is False


def test_silencio_anormal_true_when_silence_exceeds_expected() -> None:
    audio = np.zeros(int(3.0 * SAMPLE_RATE), dtype=np.float32)
    assert silencio_anormal(audio, SAMPLE_RATE, expected_pause_seconds=0.0) is True


def test_silencio_anormal_false_when_silence_matches_expected_pause() -> None:
    speech = _tone(1.0)
    silence = np.zeros(int(0.2 * SAMPLE_RATE), dtype=np.float32)
    audio = np.concatenate([speech, silence, speech])
    assert silencio_anormal(audio, SAMPLE_RATE, expected_pause_seconds=0.2) is False


def test_locutor_suspeito_true_below_threshold() -> None:
    assert locutor_suspeito(0.40) is True


def test_locutor_suspeito_false_above_threshold() -> None:
    assert locutor_suspeito(0.90) is False


def test_evaluate_segment_assembles_all_flags() -> None:
    flags = evaluate_segment(
        achieved_seconds=10.0,
        target_seconds=10.0,
        on_screen=True,
        cer=0.05,
        fidelity_score=0.95,
        audio=_tone(1.0),
        sample_rate=SAMPLE_RATE,
        expected_pause_seconds=0.0,
        speaker_similarity=0.9,
    )
    assert flags == QualityFlags(
        fora_duracao=False,
        cer_alto=False,
        traducao_infiel=False,
        clipping=False,
        silencio_anormal=False,
        locutor_suspeito=False,
    )


def test_quality_flags_serializes_to_json() -> None:
    flags = QualityFlags(
        fora_duracao=True,
        cer_alto=False,
        traducao_infiel=False,
        clipping=True,
        silencio_anormal=False,
        locutor_suspeito=False,
    )
    parsed = json.loads(flags.to_json())
    assert parsed == {
        "fora_duracao": True,
        "cer_alto": False,
        "traducao_infiel": False,
        "clipping": True,
        "silencio_anormal": False,
        "locutor_suspeito": False,
    }
