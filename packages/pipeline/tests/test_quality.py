import json

import numpy as np
import pytest

from packages.pipeline.quality import (
    QualityFlags,
    cer_alto,
    character_error_rate,
    clipping,
    evaluate_segment,
    fora_duracao,
    has_repetition,
    locutor_suspeito,
    silencio_anormal,
    traducao_infiel,
)

SAMPLE_RATE = 16000


def _tone(seconds: float, amplitude: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_character_error_rate_zero_for_identical_text() -> None:
    assert character_error_rate("ola mundo", "ola mundo") == 0.0


def test_character_error_rate_full_for_completely_different_text() -> None:
    assert character_error_rate("a", "b") == pytest.approx(1.0)


def test_character_error_rate_partial_for_one_edit() -> None:
    assert character_error_rate("teste", "testa") == pytest.approx(1 / 5)


def test_character_error_rate_empty_reference_and_hypothesis_is_zero() -> None:
    assert character_error_rate("", "") == 0.0


def test_character_error_rate_empty_reference_nonempty_hypothesis_is_one() -> None:
    assert character_error_rate("", "algo") == 1.0


def test_character_error_rate_ignores_case() -> None:
    assert character_error_rate("Já dei o troco", "já dei o troco") == 0.0


def test_character_error_rate_ignores_punctuation() -> None:
    assert character_error_rate("Olá, mundo!", "Olá mundo") == 0.0


def test_character_error_rate_collapses_whitespace() -> None:
    assert character_error_rate("isso   é    bom", "isso é bom") == 0.0


def test_character_error_rate_treats_digits_and_written_numbers_as_equal() -> None:
    assert character_error_rate("Em 2026 aconteceu.", "em dois mil e vinte e seis aconteceu") == 0.0


def test_character_error_rate_still_catches_real_differences_after_normalization() -> None:
    # Não é só normalização escondendo tudo — erro de conteúdo real continua contando.
    cer = character_error_rate("Já dei o troco pra muitos na vida.", "Já dei o troco pra ninguém.")
    assert cer > 0.15


def test_has_repetition_true_for_repeated_phrase() -> None:
    text = "Mas toma nota, mas toma nota, mas toma nota"
    assert has_repetition(text) is True


def test_has_repetition_true_for_repeated_sentence() -> None:
    text = "Já dei o troco pra muitos na vida. Já dei o troco pra muitos na vida."
    assert has_repetition(text) is True


def test_has_repetition_false_for_normal_sentence() -> None:
    text = "Tudo que eles deram pra isso tá valendo zero"
    assert has_repetition(text) is False


def test_has_repetition_false_for_short_text() -> None:
    assert has_repetition("Mas toma nota.") is False


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
