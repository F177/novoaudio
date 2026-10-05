import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from packages.pipeline.quality import (
    QualityFlags,
    cer_alto,
    character_error_rate,
    clipping,
    count_energy_bursts,
    evaluate_segment,
    fora_duracao,
    has_repetition,
    locutor_suspeito,
    loop_confirmed_by_bursts,
    silencio_anormal,
    traducao_infiel,
    truncado,
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


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def test_count_energy_bursts_counts_separate_voiced_regions() -> None:
    audio = np.concatenate(
        [_silence(0.2), _tone(0.3), _silence(0.2), _tone(0.3), _silence(0.2), _tone(0.3)]
    )
    assert count_energy_bursts(audio, SAMPLE_RATE) == 3


def test_count_energy_bursts_single_continuous_voiced_region_is_one() -> None:
    audio = np.concatenate([_silence(0.1), _tone(1.0), _silence(0.1)])
    assert count_energy_bursts(audio, SAMPLE_RATE) == 1


def test_count_energy_bursts_all_silence_is_zero() -> None:
    assert count_energy_bursts(_silence(1.0), SAMPLE_RATE) == 0


def test_count_energy_bursts_matches_documented_seg0009_ground_truth() -> None:
    # Regressão real (T0.19): docs/moss_tts_investigation.md registra 5-6
    # rajadas contadas à mão nesse áudio exato (1,84s, referência "Deixa as
    # bobagens pra lá.", 5 palavras) — o caso que expôs a alucinação de
    # repetição do Whisper (transcreveu a frase 28x num áudio sem loop
    # real nenhum). Fixture committada de propósito (ver .gitignore) pra
    # amarrar a função ao achado documentado, não a dado sintético.
    fixture_path = Path(__file__).parent / "fixtures" / "seg0009_base.wav"
    audio, sr = sf.read(fixture_path, dtype="float32")
    bursts = count_energy_bursts(audio, sr)
    assert 5 <= bursts <= 6


def test_loop_confirmed_by_bursts_true_when_far_above_expected() -> None:
    # dado real do teste de mutação da T0.19 (frase de 10 palavras colada
    # 2x): ~12 rajadas numa cópia, ~24 em duas — 24 > 10×1,5 confirma loop
    assert loop_confirmed_by_bursts(burst_count=24, expected_words=10) is True


def test_loop_confirmed_by_bursts_false_when_matching_expected() -> None:
    # mesmo caso do seg0009: 5 rajadas reais, 5 palavras esperadas — não é
    # loop, mesmo que o Whisper tenha alucinado repetição na transcrição
    assert loop_confirmed_by_bursts(burst_count=5, expected_words=5) is False


def test_loop_confirmed_by_bursts_single_copy_does_not_false_positive() -> None:
    # dado real: 10 palavras, ~12 rajadas numa cópia só (sem duplicar) —
    # não pode disparar, ou toda frase normal seria flagrada como loop
    assert loop_confirmed_by_bursts(burst_count=12, expected_words=10) is False


def test_loop_confirmed_by_bursts_zero_expected_words_any_burst_is_suspect() -> None:
    assert loop_confirmed_by_bursts(burst_count=1, expected_words=0) is True
    assert loop_confirmed_by_bursts(burst_count=0, expected_words=0) is False


def test_truncado_true_when_tail_of_reference_is_missing() -> None:
    # T0.14: fora_duracao não pega isso por construção — o MOSS-TTS corta o
    # texto pra caber na duração pedida, então a duração bate mesmo cortado.
    reference = "Tudo que eles deram pra isso tá valendo zero"
    hypothesis = "Tudo que eles deram pra isso"
    assert truncado(reference, hypothesis) is True


def test_truncado_false_when_content_matches() -> None:
    reference = "Eles tinham força pra cima de todos nós juntos."
    hypothesis = "Eles tinham força pra cima de todos nós juntos."
    assert truncado(reference, hypothesis) is False


def test_truncado_false_for_short_reference() -> None:
    # Critério de aceite explícito do T0.14: referência curta (<4 palavras)
    # não deve disparar, mesmo com a hipótese vazia (que pareceria "cortada").
    assert truncado("Mas toma nota.", "") is False


def test_truncado_false_for_minor_asr_paraphrase() -> None:
    # Diferença de transcrição normal (contração lida por extenso), não corte.
    reference = "Eles tá morrendo enfrentando inimigos."
    hypothesis = "Ele está morrendo, enfrentando inimigos."
    assert truncado(reference, hypothesis) is False


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
        reference_text="Eles tinham força pra cima de todos nós juntos.",
        hypothesis_text="Eles tinham força pra cima de todos nós juntos.",
        fidelity_score=0.95,
        audio=_tone(1.0),
        sample_rate=SAMPLE_RATE,
        expected_pause_seconds=0.0,
        speaker_similarity=0.9,
    )
    assert flags == QualityFlags(
        fora_duracao=False,
        cer_alto=False,
        truncado=False,
        traducao_infiel=False,
        clipping=False,
        silencio_anormal=False,
        locutor_suspeito=False,
    )


def test_quality_flags_serializes_to_json() -> None:
    flags = QualityFlags(
        fora_duracao=True,
        cer_alto=False,
        truncado=False,
        traducao_infiel=False,
        clipping=True,
        silencio_anormal=False,
        locutor_suspeito=False,
    )
    parsed = json.loads(flags.to_json())
    assert parsed == {
        "fora_duracao": True,
        "cer_alto": False,
        "truncado": False,
        "traducao_infiel": False,
        "clipping": True,
        "silencio_anormal": False,
        "locutor_suspeito": False,
        "entonacao_desalinhada": False,
        "emocao_incompativel": False,
        "cer_final_alto": False,
        "loop_confirmado": False,
    }
