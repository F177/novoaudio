import numpy as np
import pytest

from packages.pipeline.segmentation import InternalPause
from packages.pipeline.synthesis import (
    MAX_TIME_STRETCH_RATIO,
    MIN_SYNTHESIS_TOKENS,
    adjust_tokens_for_retry,
    apply_ipa_overrides,
    duration_to_tokens,
    infer_delivery_instruction,
    inject_pauses,
    is_within_tolerance,
    time_stretch_to_duration,
)

SAMPLE_RATE = 24000


def _tone(seconds: float, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_duration_to_tokens_uses_12_5hz() -> None:
    assert duration_to_tokens(8.0) == 100


def test_duration_to_tokens_includes_pause_time() -> None:
    # a duracao alvo ja deve incluir o tempo de pausa planejado (ver docstring)
    with_pause_budget = duration_to_tokens(10.0)
    without_pause_budget = duration_to_tokens(8.0)
    assert with_pause_budget > without_pause_budget


def test_duration_to_tokens_never_goes_below_moss_tts_delay_pattern_floor() -> None:
    # T0.9/investigação de 2026-09-14: abaixo de MIN_SYNTHESIS_TOKENS o
    # "delay pattern" (n_vq=32 codebooks) nunca completa um ciclo — o
    # segmento sai vazio/garbled, não importa quão curto seja o alvo real.
    assert duration_to_tokens(0.3) == MIN_SYNTHESIS_TOKENS
    assert duration_to_tokens(1.0) == MIN_SYNTHESIS_TOKENS


def test_duration_to_tokens_above_floor_is_unaffected() -> None:
    # 8s * 12.5 = 100 tokens, bem acima do piso — não deve ser clampado.
    assert duration_to_tokens(8.0) > MIN_SYNTHESIS_TOKENS


def test_inject_pauses_no_pauses_returns_text_unchanged() -> None:
    text = "Isso e um teste."
    assert inject_pauses(text, [], original_word_count=3) == text


def test_inject_pauses_places_marker_at_relative_position() -> None:
    # pausa depois da 2a de 4 palavras originais (50% do caminho)
    pausas = [InternalPause(after_word_index=1, duration=0.5)]
    translated = "Um dois tres quatro"
    result = inject_pauses(translated, pausas, original_word_count=4)
    assert "[pause 0.5s]" in result


def test_inject_pauses_marker_format() -> None:
    pausas = [InternalPause(after_word_index=0, duration=1.25)]
    result = inject_pauses("Ola mundo", pausas, original_word_count=2)
    assert "[pause 1.2s]" in result or "[pause 1.3s]" in result


def test_inject_pauses_multiple_pauses_keep_order() -> None:
    pausas = [
        InternalPause(after_word_index=0, duration=0.3),
        InternalPause(after_word_index=2, duration=0.5),
    ]
    translated = "A B C D"
    result = inject_pauses(translated, pausas, original_word_count=4)
    assert result.index("[pause 0.3s]") < result.index("[pause 0.5s]")


def test_apply_ipa_overrides_replaces_term() -> None:
    text = "Eu uso OpenAI todo dia."
    result = apply_ipa_overrides(text, {"OpenAI": "oʊˈpɛn aɪ"})
    assert "/oʊˈpɛn aɪ/" in result
    assert "OpenAI" not in result


def test_apply_ipa_overrides_is_case_insensitive() -> None:
    text = "eu uso openai todo dia."
    result = apply_ipa_overrides(text, {"OpenAI": "oʊˈpɛn aɪ"})
    assert "/oʊˈpɛn aɪ/" in result


def test_apply_ipa_overrides_leaves_text_unchanged_without_matches() -> None:
    text = "Nada a ver com o glossario aqui."
    assert apply_ipa_overrides(text, {"OpenAI": "oʊˈpɛn aɪ"}) == text


def test_is_within_tolerance() -> None:
    assert is_within_tolerance(10.5, 10.0, tolerance=0.08) is True
    assert is_within_tolerance(12.0, 10.0, tolerance=0.08) is False


def test_adjust_tokens_for_retry_corrects_proportionally() -> None:
    # sintese anterior: 100 tokens gerou 10s (1s a menos que os 8.0 nominal
    # esperaria... aqui simulamos taxa real de 0.1s/token em vez de 0.08)
    new_tokens = adjust_tokens_for_retry(
        current_tokens=100, achieved_seconds=10.0, target_seconds=8.0
    )
    assert new_tokens == 80


def test_adjust_tokens_for_retry_falls_back_when_achieved_is_zero() -> None:
    assert adjust_tokens_for_retry(100, 0.0, 8.0) == duration_to_tokens(8.0)


def test_adjust_tokens_for_retry_never_returns_zero_or_negative() -> None:
    result = adjust_tokens_for_retry(
        current_tokens=100, achieved_seconds=1000.0, target_seconds=0.01
    )
    assert result >= 1


def test_adjust_tokens_for_retry_never_goes_below_moss_tts_delay_pattern_floor() -> None:
    # Um segmento curto que já falhou por estar abaixo do piso não deve
    # pedir tokens ainda mais baixos na tentativa seguinte.
    result = adjust_tokens_for_retry(
        current_tokens=100, achieved_seconds=1000.0, target_seconds=0.01
    )
    assert result == MIN_SYNTHESIS_TOKENS


def test_time_stretch_to_duration_compresses_to_target() -> None:
    audio = _tone(3.2)
    stretched = time_stretch_to_duration(audio, SAMPLE_RATE, target_seconds=2.4)
    assert len(stretched) / SAMPLE_RATE == pytest.approx(2.4, abs=0.05)


def test_time_stretch_to_duration_stretches_to_target() -> None:
    audio = _tone(1.0)
    stretched = time_stretch_to_duration(audio, SAMPLE_RATE, target_seconds=1.4)
    assert len(stretched) / SAMPLE_RATE == pytest.approx(1.4, abs=0.05)


def test_time_stretch_to_duration_noop_when_already_close() -> None:
    audio = _tone(2.0)
    stretched = time_stretch_to_duration(audio, SAMPLE_RATE, target_seconds=2.0)
    assert np.array_equal(stretched, audio)


def test_time_stretch_to_duration_clamps_extreme_ratio() -> None:
    # 3.2s -> 0.4s pediria 8x de compressão, muito acima do que soa
    # reconhecível — trava em MAX_TIME_STRETCH_RATIO e chega o mais perto
    # possível, não exatamente no alvo.
    audio = _tone(3.2)
    stretched = time_stretch_to_duration(audio, SAMPLE_RATE, target_seconds=0.4)
    achieved = len(stretched) / SAMPLE_RATE
    expected_capped = 3.2 / MAX_TIME_STRETCH_RATIO
    assert achieved == pytest.approx(expected_capped, abs=0.05)
    assert achieved > 0.4  # não chegou no alvo real, ficou no limite seguro


def test_time_stretch_to_duration_respects_custom_max_ratio() -> None:
    audio = _tone(3.2)
    stretched = time_stretch_to_duration(audio, SAMPLE_RATE, target_seconds=0.4, max_ratio=10.0)
    achieved = len(stretched) / SAMPLE_RATE
    assert achieved == pytest.approx(0.4, abs=0.05)


def test_infer_delivery_instruction_exclamation() -> None:
    assert infer_delivery_instruction("Corre agora!") is not None


def test_infer_delivery_instruction_question() -> None:
    instruction = infer_delivery_instruction("Você vem comigo?")
    assert instruction is not None
    assert "pergunta" in instruction


def test_infer_delivery_instruction_exclamation_takes_priority_over_question() -> None:
    exclamation_instruction = infer_delivery_instruction("Corre, agora!")
    question_instruction = infer_delivery_instruction("Você vem?")
    assert exclamation_instruction != question_instruction


def test_infer_delivery_instruction_neutral_text_returns_none() -> None:
    assert infer_delivery_instruction("Eles foram embora ontem à noite.") is None
