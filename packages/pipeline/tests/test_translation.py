import json

import pytest

from packages.pipeline.translation import (
    build_prompt,
    extract_proper_nouns,
    parse_candidates,
    rank_candidates,
    score_candidate,
)


def test_build_prompt_includes_source_text_and_budget() -> None:
    prompt = build_prompt("The dog ran fast.", target_syllables=12)
    assert "The dog ran fast." in prompt
    assert "12 s" in prompt or "12 sílabas" in prompt


def test_build_prompt_mentions_spoken_ptbr_levers() -> None:
    prompt = build_prompt("Hello.", target_syllables=3)
    assert "pro-drop" in prompt
    assert "voz ativa" in prompt


def test_build_prompt_asks_to_preserve_emphasis_punctuation() -> None:
    prompt = build_prompt("Hello!", target_syllables=3)
    assert "ênfase" in prompt


def test_build_prompt_forbids_chat_abbreviations_but_keeps_spoken_contractions() -> None:
    prompt = build_prompt("Hello.", target_syllables=3)
    assert "abreviação de chat" in prompt
    assert '"tá"' in prompt


def test_build_prompt_without_context_omits_context_block() -> None:
    prompt = build_prompt("Hello.", target_syllables=3)
    assert "Contexto" not in prompt


def test_build_prompt_includes_neighbor_context_but_marks_it_not_to_translate() -> None:
    prompt = build_prompt(
        "Hello.",
        target_syllables=3,
        previous_text="He ran fast.",
        next_text="She won.",
    )
    assert "He ran fast." in prompt
    assert "She won." in prompt
    assert "NÃO traduza" in prompt


def test_build_prompt_includes_preserve_terms() -> None:
    prompt = build_prompt("Hello.", target_syllables=3, preserve_terms=["Hurts", "NFL"])
    assert "Hurts" in prompt
    assert "NFL" in prompt


def test_build_prompt_without_preserve_terms_omits_preserve_block() -> None:
    prompt = build_prompt("Hello.", target_syllables=3)
    assert "preserve" not in prompt.lower() or "preserve_terms" not in prompt


def test_build_prompt_without_continuation_omits_continuation_instruction() -> None:
    prompt = build_prompt("Hello.", target_syllables=3)
    assert "EM MEIO a uma frase" not in prompt


def test_build_prompt_with_continuation_warns_not_to_drop_content() -> None:
    prompt = build_prompt(
        "for that. What are you talking about?", target_syllables=10, is_continuation=True
    )
    assert "EM MEIO a uma frase" in prompt
    assert "não corte" in prompt.lower() or "nao corte" in prompt.lower()


def test_extract_proper_nouns_catches_non_sentence_initial_capitalized_word() -> None:
    result = extract_proper_nouns(["I saw Marcus yesterday."])
    assert "Marcus" in result


def test_extract_proper_nouns_catches_acronym() -> None:
    result = extract_proper_nouns(["The NFL season starts today."])
    assert "NFL" in result


def test_extract_proper_nouns_catches_name_repeated_sentence_initial_across_segments() -> None:
    # "Hurts" (sobrenome do quarterback) só aparece no início de frase, mas em
    # dois segmentos diferentes — achado real do T0.23 (ver TASKS.md).
    result = extract_proper_nouns(["Hurts has time.", "Hurts threw deep."])
    assert "Hurts" in result


def test_extract_proper_nouns_ignores_common_sentence_starters() -> None:
    result = extract_proper_nouns(["The game is close.", "The crowd roars."])
    assert "The" not in result


def test_extract_proper_nouns_ignores_single_occurrence_sentence_initial_word() -> None:
    # Ambíguo (pode ser nome próprio citado uma vez só, ou palavra comum que
    # calha de abrir a única frase) — a função prefere não marcar, critério
    # de aceite da T0.23 é checagem manual, não recall perfeito. Limitação
    # conhecida: um nome próprio citado uma única vez, só no início de
    # frase, escapa desta heurística.
    result = extract_proper_nouns(["Marcus ran fast."])
    assert "Marcus" not in result


def test_extract_proper_nouns_dedupes_preserving_first_spelling() -> None:
    result = extract_proper_nouns(["I met Marcus.", "Marcus again, I met Marcus."])
    assert result.count("Marcus") == 1


def test_parse_candidates_extracts_list() -> None:
    response = json.dumps({"candidates": ["Oi.", "Olá, tudo bem?"]})
    assert parse_candidates(response) == ["Oi.", "Olá, tudo bem?"]


def test_parse_candidates_tolerates_markdown_wrapping() -> None:
    response = '```json\n{"candidates": ["Oi.", "Ola."]}\n```'
    assert parse_candidates(response) == ["Oi.", "Ola."]


def test_parse_candidates_tolerates_prose_before_and_after() -> None:
    response = 'Aqui estao as opcoes:\n{"candidates": ["Um.", "Dois."]}\nEspero que ajude!'
    assert parse_candidates(response) == ["Um.", "Dois."]


def test_parse_candidates_raises_without_json() -> None:
    with pytest.raises(ValueError, match="JSON"):
        parse_candidates("desculpa, nao consegui gerar candidatos")


def test_parse_candidates_raises_without_candidates_key() -> None:
    with pytest.raises(ValueError, match="candidates"):
        parse_candidates(json.dumps({"outra_coisa": []}))


def test_parse_candidates_unwraps_single_element_nested_list() -> None:
    # Achado real (beast.mp4, 2026-09-18): o Qwen às vezes aninha cada
    # candidato numa lista de 1 elemento em vez de string solta — e faz
    # isso de forma repetida/consistente pro mesmo segmento, não por acaso
    # de amostragem (visto 3 tentativas seguidas com o mesmo padrão).
    # Aceitar isso cegamente (`str(c)`) mandava o texto literal "['...']"
    # pro MOSS-TTS sintetizar; rejeitar e torcer pelo retry desperdiçava
    # tentativas à toa quando o formato é conhecido e recuperável.
    response = json.dumps({"candidates": [["Oi, tudo bem?"], "Olá."]})
    assert parse_candidates(response) == ["Oi, tudo bem?", "Olá."]


def test_parse_candidates_raises_when_nested_list_is_not_recoverable() -> None:
    # Lista com mais de 1 item, ou item que não é string — não é o padrão
    # exato conhecido, continua sendo rejeitado (não tenta adivinhar).
    response = json.dumps({"candidates": [["Oi.", "Tchau."], "Olá."]})
    with pytest.raises(ValueError, match="não é uma string"):
        parse_candidates(response)


def test_parse_candidates_raises_on_leftover_list_bracket_residue() -> None:
    # Achado real (T0.22, lego_batman seg0002): mesmo desaninhado, a
    # string em si chegou ao TTS começando com "['" — colchete e aspas
    # falados pelo MOSS-TTS, sem validação nenhuma. Diferente do caso de
    # `[["texto"]]` acima (padrão exato, recuperável), aqui o resíduo fica
    # DENTRO da string, sem jeito óbvio de limpar — rejeita pro retry.
    response = json.dumps({"candidates": ["['Alguma fala aqui", "Olá."]})
    with pytest.raises(ValueError, match="resíduo de estrutura"):
        parse_candidates(response)


def test_parse_candidates_raises_on_leading_quote_residue() -> None:
    response = json.dumps({"candidates": ["'Alguma fala", "Olá."]})
    with pytest.raises(ValueError, match="resíduo de estrutura"):
        parse_candidates(response)


def test_parse_candidates_allows_text_with_bracket_in_the_middle() -> None:
    # só o INÍCIO da string é suspeito — colchete/aspas no meio de uma
    # fala real (ex.: citação) não deve ser rejeitado
    response = json.dumps({"candidates": ["Ele disse 'oi' pra mim", "Olá."]})
    assert parse_candidates(response) == ["Ele disse 'oi' pra mim", "Olá."]


def test_parse_candidates_ignores_extra_braces_after_the_real_json() -> None:
    # Achado real: o regex guloso antigo (`\{.*\}` com DOTALL) ia do
    # primeiro `{` até o ÚLTIMO `}` da resposta inteira — se o LLM
    # escrevesse qualquer chave extra depois do JSON de verdade (exemplo,
    # rascunho, nota), o match antigo engolia tudo junto. A extração por
    # contagem de chaves para no par certo do primeiro `{`.
    response = (
        '{"candidates": ["Oi.", "Ola."]}\n'
        'Nota: o formato de exemplo seria algo como {"candidates": ["exemplo"]}'
    )
    assert parse_candidates(response) == ["Oi.", "Ola."]


def test_parse_candidates_raises_on_unclosed_json() -> None:
    with pytest.raises(ValueError, match="incompleto"):
        parse_candidates('{"candidates": ["Oi.", "trunc')


def test_score_candidate_prefers_matching_budget() -> None:
    exact = score_candidate(
        "Uma frase com doze silabas mais ou menos aqui", target_syllables=12, source_text=""
    )
    way_off = score_candidate("Oi", target_syllables=12, source_text="")
    assert exact.budget_score > way_off.budget_score


def test_score_candidate_detects_spoken_markers() -> None:
    natural = score_candidate("Cê tá indo pra casa?", target_syllables=6, source_text="")
    formal = score_candidate(
        "Você está indo para a residência?", target_syllables=8, source_text=""
    )
    assert natural.naturalness_score > formal.naturalness_score


def test_score_candidate_rewards_matching_emphasis_punctuation() -> None:
    matching = score_candidate("Corre!", target_syllables=2, source_text="Run!")
    flattened = score_candidate("Corre.", target_syllables=2, source_text="Run!")
    assert matching.expressiveness_score > flattened.expressiveness_score


def test_score_candidate_penalizes_adding_emphasis_not_in_source() -> None:
    flat_source = score_candidate("Ele foi embora!", target_syllables=5, source_text="He left.")
    matching = score_candidate("Ele foi embora.", target_syllables=5, source_text="He left.")
    assert matching.expressiveness_score > flat_source.expressiveness_score


def test_rank_candidates_orders_best_first() -> None:
    candidates = [
        "Isso e uma frase bem mais longa do que o orcamento pedia originalmente",
        "Isso cabe certinho",
        "Oi",
    ]
    ranked = rank_candidates(candidates, target_syllables=6, source_text="")
    assert ranked[0].text == "Isso cabe certinho"
    assert ranked == sorted(ranked, key=lambda c: c.score, reverse=True)


def test_rank_candidates_returns_one_result_per_candidate() -> None:
    candidates = ["A.", "B.", "C.", "D.", "E."]
    ranked = rank_candidates(candidates, target_syllables=1, source_text="")
    assert len(ranked) == 5


def test_rank_candidates_prefers_preserved_emphasis_when_budget_ties() -> None:
    candidates = ["Corre.", "Corre!"]
    ranked = rank_candidates(candidates, target_syllables=2, source_text="Run!")
    assert ranked[0].text == "Corre!"


def test_score_candidate_continuation_does_not_penalize_going_over_budget() -> None:
    # T0.23+ achado real: pra fragmento de frase cortada, um candidato mais
    # longo que o orçamento (preservando conteúdo) não deve perder pontos
    # de orçamento — o sistema comprime áudio depois, mas não recupera
    # conteúdo que a tradução já cortou.
    over_budget = score_candidate(
        "Uma frase bem mais longa do que os quatro silabas pedidas mas completa",
        target_syllables=4,
        source_text="",
        is_continuation=True,
    )
    assert over_budget.budget_score == 1.0


def test_score_candidate_continuation_still_penalizes_going_under_budget() -> None:
    # Só o lado "mais longo" é perdoado — ficar mais CURTO que o orçamento
    # ainda é penalizado (pode ser sinal de conteúdo cortado de outro jeito).
    under_budget = score_candidate("Oi", target_syllables=12, source_text="", is_continuation=True)
    assert under_budget.budget_score < 1.0


def test_score_candidate_non_continuation_keeps_symmetric_penalty() -> None:
    # Sem is_continuation (default), comportamento idêntico ao antigo —
    # ficar mais longo que o orçamento ainda é penalizado normalmente.
    over_budget = score_candidate(
        "Uma frase bem mais longa do que os quatro silabas pedidas",
        target_syllables=4,
        source_text="",
    )
    assert over_budget.budget_score < 1.0


def test_rank_candidates_continuation_prefers_complete_over_short() -> None:
    # Cenário real (seg0010, vídeo padrão): candidato curto que bate o
    # orçamento mas perdeu conteúdo ("for that") perdia pro completo antes
    # da mudança de pontuação pra fragmentos. "Que isso" (3 sílabas, fica
    # abaixo do orçamento) simula o candidato que cortou "for that"; "Pra
    # isso, que isso" (6 sílabas, no orçamento) simula o que preservou.
    short_incomplete = "Que isso"
    long_complete = "Pra isso, que isso"
    ranked = rank_candidates(
        [short_incomplete, long_complete],
        target_syllables=6,
        source_text="",
        is_continuation=True,
    )
    assert ranked[0].text == long_complete
