import json

import pytest

from packages.pipeline.translation import (
    PT_BR,
    TargetLanguage,
    WindowLine,
    build_judge_prompt,
    build_shorten_prompt,
    build_window_prompt,
    build_windows,
    choose_alternatives,
    choose_shortened,
    contracted,
    extract_proper_nouns,
    has_disallowed_script,
    has_forbidden_word,
    mt_options,
    parse_candidates,
    parse_window_response,
    profanity_count,
    rank_candidates,
    score_candidate,
    sentence_groups,
    split_by_alignment,
    split_by_source,
    split_sentences,
    split_window,
)


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
    assert parse_candidates(response, PT_BR) == ["Oi.", "Olá, tudo bem?"]


def test_parse_candidates_tolerates_markdown_wrapping() -> None:
    response = '```json\n{"candidates": ["Oi.", "Ola."]}\n```'
    assert parse_candidates(response, PT_BR) == ["Oi.", "Ola."]


def test_parse_candidates_tolerates_prose_before_and_after() -> None:
    response = 'Aqui estao as opcoes:\n{"candidates": ["Um.", "Dois."]}\nEspero que ajude!'
    assert parse_candidates(response, PT_BR) == ["Um.", "Dois."]


def test_parse_candidates_raises_without_json() -> None:
    with pytest.raises(ValueError, match="JSON"):
        parse_candidates("desculpa, nao consegui gerar candidatos", PT_BR)


def test_parse_candidates_raises_without_candidates_key() -> None:
    with pytest.raises(ValueError, match="candidates"):
        parse_candidates(json.dumps({"outra_coisa": []}), PT_BR)


def test_parse_candidates_unwraps_single_element_nested_list() -> None:
    # Achado real (beast.mp4, 2026-09-18): o Qwen às vezes aninha cada
    # candidato numa lista de 1 elemento em vez de string solta — e faz
    # isso de forma repetida/consistente pro mesmo segmento, não por acaso
    # de amostragem (visto 3 tentativas seguidas com o mesmo padrão).
    # Aceitar isso cegamente (`str(c)`) mandava o texto literal "['...']"
    # pro MOSS-TTS sintetizar; rejeitar e torcer pelo retry desperdiçava
    # tentativas à toa quando o formato é conhecido e recuperável.
    response = json.dumps({"candidates": [["Oi, tudo bem?"], "Olá."]})
    assert parse_candidates(response, PT_BR) == ["Oi, tudo bem?", "Olá."]


def test_parse_candidates_raises_when_nested_list_is_not_recoverable() -> None:
    # Lista com mais de 1 item, ou item que não é string — não é o padrão
    # exato conhecido, continua sendo rejeitado (não tenta adivinhar).
    response = json.dumps({"candidates": [["Oi.", "Tchau."], "Olá."]})
    with pytest.raises(ValueError, match="não é uma string"):
        parse_candidates(response, PT_BR)


def test_parse_candidates_raises_on_leftover_list_bracket_residue() -> None:
    # Achado real (T0.22, lego_batman seg0002): mesmo desaninhado, a
    # string em si chegou ao TTS começando com "['" — colchete e aspas
    # falados pelo MOSS-TTS, sem validação nenhuma. Diferente do caso de
    # `[["texto"]]` acima (padrão exato, recuperável), aqui o resíduo fica
    # DENTRO da string, sem jeito óbvio de limpar — rejeita pro retry.
    response = json.dumps({"candidates": ["['Alguma fala aqui", "Olá."]})
    with pytest.raises(ValueError, match="resíduo de estrutura"):
        parse_candidates(response, PT_BR)


def test_parse_candidates_raises_on_leading_quote_residue() -> None:
    response = json.dumps({"candidates": ["'Alguma fala", "Olá."]})
    with pytest.raises(ValueError, match="resíduo de estrutura"):
        parse_candidates(response, PT_BR)


def test_parse_candidates_allows_text_with_bracket_in_the_middle() -> None:
    # só o INÍCIO da string é suspeito — colchete/aspas no meio de uma
    # fala real (ex.: citação) não deve ser rejeitado
    response = json.dumps({"candidates": ["Ele disse 'oi' pra mim", "Olá."]})
    assert parse_candidates(response, PT_BR) == ["Ele disse 'oi' pra mim", "Olá."]


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
    assert parse_candidates(response, PT_BR) == ["Oi.", "Ola."]


def test_parse_candidates_raises_on_unclosed_json() -> None:
    with pytest.raises(ValueError, match="incompleto"):
        parse_candidates('{"candidates": ["Oi.", "trunc', PT_BR)


def test_parse_candidates_drops_candidate_that_switches_to_chinese() -> None:
    # Achado real (NFL seg0022): "Ele foi derrubado, mas显而易见".
    response = json.dumps(
        {"candidates": ["Ele foi derrubado, mas显而易见", "Ele caiu, mas claramente não."]},
        ensure_ascii=False,
    )
    assert parse_candidates(response, PT_BR) == ["Ele caiu, mas claramente não."]


def test_parse_candidates_raises_when_every_candidate_has_foreign_script() -> None:
    response = json.dumps({"candidates": ["mas显而易见", "Привет"]}, ensure_ascii=False)
    with pytest.raises(ValueError, match="alfabeto latino"):
        parse_candidates(response, PT_BR)


def test_parse_candidates_keeps_portuguese_accents_and_ordinals() -> None:
    text = "Ação, coração, pé, à, você, ü, 1º lugar e 2ª vez."
    response = json.dumps({"candidates": [text]}, ensure_ascii=False)
    assert parse_candidates(response, PT_BR) == [text]


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


def _lines(n: int) -> list[WindowLine]:
    return [WindowLine(f"seg{i:04d}", "A" if i % 2 else "B", f"line {i}") for i in range(n)]


def test_build_windows_covers_every_line_once_in_order() -> None:
    windows = build_windows(_lines(30), size=12, context_lines=3)
    flat = [line.id for _, window in windows for line in window]
    assert flat == [line.id for line in _lines(30)]
    assert [len(w) for _, w in windows] == [12, 12, 6]


def test_build_windows_gives_previous_lines_as_context() -> None:
    windows = build_windows(_lines(30), size=12, context_lines=3)
    assert windows[0][0] == []
    assert [line.id for line in windows[1][0]] == ["seg0009", "seg0010", "seg0011"]


def test_window_prompt_contains_lines_speakers_and_keep_content_rules() -> None:
    prompt = build_window_prompt(
        [WindowLine("seg0003", "B", "Here you"), WindowLine("seg0004", "A", "go. Hello!")],
        PT_BR,
        context_before=[WindowLine("seg0002", "A", "Call your dad.")],
        preserve_terms=["NFL"],
    )
    assert '"id": "seg0003"' in prompt and '"speaker": "B"' in prompt
    assert "Call your dad." in prompt and "do NOT translate these" in prompt
    assert "NFL" in prompt
    assert "profanity" in prompt and "change the content" in prompt
    assert "Brazilian Portuguese" in prompt and '"vcs"' in prompt


def test_window_prompt_omits_optional_blocks() -> None:
    prompt = build_window_prompt([WindowLine("seg0000", "A", "Hi.")], PT_BR)
    assert "do NOT translate these" not in prompt
    assert "keep exactly as written" not in prompt


def test_parse_window_response_maps_ids_to_text() -> None:
    response = json.dumps(
        {"translations": [{"id": "seg0001", "text": "Toma."}, {"id": "seg0002", "text": "Vai!"}]},
        ensure_ascii=False,
    )
    lines = [WindowLine("seg0001", "A", "Take it."), WindowLine("seg0002", "B", "Go!")]
    assert parse_window_response(response, lines, PT_BR) == {
        "seg0001": "Toma.",
        "seg0002": "Vai!",
    }


def test_parse_window_response_tolerates_prose_around_json() -> None:
    response = 'Sure: {"translations": [{"id": "seg0001", "text": "Oi."}]} done.'
    lines = [WindowLine("seg0001", "A", "Hi.")]
    assert parse_window_response(response, lines, PT_BR) == {"seg0001": "Oi."}


@pytest.mark.parametrize(
    "items",
    [
        [{"id": "seg0001", "text": "Oi."}],  # faltando seg0002
        [
            {"id": "seg0001", "text": "Oi."},
            {"id": "seg0002", "text": "A"},
            {"id": "x", "text": "B"},
        ],
        [{"id": "seg0001", "text": "Oi."}, {"id": "seg0001", "text": "De novo."}],
        [{"id": "seg0001", "text": "Oi."}, {"id": "seg0002", "text": "  "}],
        [{"id": "seg0001", "text": "Oi."}, {"id": "seg0002", "text": "mas显而易见"}],
    ],
)
def test_parse_window_response_rejects_misaligned_or_bad_output(items) -> None:
    response = json.dumps({"translations": items}, ensure_ascii=False)
    with pytest.raises(ValueError):
        parse_window_response(
            response, [WindowLine("seg0001", "A", "Hi."), WindowLine("seg0002", "B", "Hi.")], PT_BR
        )


def test_language_rules_come_from_the_profile_not_the_shared_prompt() -> None:
    # Portabilidade: um idioma novo é só um perfil novo; nada de pt-BR no
    # prompt compartilhado.
    other = TargetLanguage(
        code="xx",
        name="Testlandish",
        scripts=("LATIN",),
        extra_letters=frozenset(),
        style_rules=("Use the polite form.",),
        char_ratio_range=(0.5, 2.0),
        count_syllables=len,
        compression_examples=(),
        forbidden_words=frozenset(),
        profanity_stems=(),
    )
    prompt = build_window_prompt([WindowLine("seg0000", "A", "Hi.")], other)
    assert "Testlandish" in prompt and "Use the polite form." in prompt
    assert "Portuguese" not in prompt and '"vcs"' not in prompt and "tá" not in prompt


def test_script_check_follows_the_target_language() -> None:
    cyrillic_target = TargetLanguage(
        code="ru",
        name="Russian",
        scripts=("CYRILLIC",),
        extra_letters=frozenset(),
        style_rules=(),
        char_ratio_range=(0.5, 2.0),
        count_syllables=len,
        compression_examples=(),
        forbidden_words=frozenset(),
        profanity_stems=(),
    )
    assert has_disallowed_script("Привет", PT_BR)
    assert not has_disallowed_script("Привет", cyrillic_target)
    assert has_disallowed_script("Hello", cyrillic_target)


def test_split_window_halves_and_carries_context() -> None:
    lines = _lines(12)
    context = [WindowLine("ctx", "A", "before")]
    (c1, first), (c2, second) = split_window(context, lines, context_lines=3)
    assert c1 == context
    assert [line.id for line in first] == [f"seg{i:04d}" for i in range(6)]
    assert [line.id for line in second] == [f"seg{i:04d}" for i in range(6, 12)]
    assert [line.id for line in c2] == ["seg0003", "seg0004", "seg0005"]


def test_split_window_down_to_single_lines_covers_everything() -> None:
    # Repetindo a divisão, toda janela chega a falas sozinhas sem perder nenhuma.
    pending = [([], _lines(12))]
    singles: list[str] = []
    while pending:
        context, lines = pending.pop()
        if len(lines) == 1:
            singles.append(lines[0].id)
        else:
            pending.extend(split_window(context, lines))
    assert sorted(singles) == [line.id for line in _lines(12)]


def test_parse_window_response_rejects_content_shifted_between_lines() -> None:
    # Caso real (Short do Jay Shetty): ids todos presentes, mas a primeira
    # fala recebeu as frases das seguintes e a última ficou só com o rabo.
    first = "Say that again, Jay. Weak people notice other people's mistakes"
    last = (
        "but they're seeing different things because no one is a weak or "
        "strong person. It's just what you focus on."
    )
    lines = [WindowLine("seg0000", "A", first), WindowLine("seg0009", "A", last)]
    response = json.dumps({"translations": [
        {"id": "seg0000", "text": "Diga de novo, Jay. Pessoas fracas notam os erros dos outros e "
         "riem. Pessoas fortes notam os erros dos outros e aprendem. E falam sobre os problemas "
         "dos outros pra se sentirem melhor."},
        {"id": "seg0009", "text": "É só o que você foca."},
    ]}, ensure_ascii=False)
    with pytest.raises(ValueError, match="tamanho"):
        parse_window_response(response, lines, PT_BR)


def test_shorten_prompt_carries_target_examples_and_keep_content_rule() -> None:
    prompt = build_shorten_prompt(
        "I don't even know how to make the payment.",
        "Eu nem sei como é que eu vou fazer o pagamento.",
        8,
        PT_BR,
    )
    assert "about 8 syllables" in prompt and "ALL that the English original says" in prompt
    assert "profanity" in prompt and "never telegraphic" in prompt
    assert "Brazilian Portuguese" in prompt


def test_choose_shortened_picks_most_faithful_shorter_candidate() -> None:
    options = [
        ("curto mas sem sentido", 4, 0.30),
        ("curto e fiel", 6, 0.93),
        ("mais fiel porém mais longo", 12, 0.97),
        ("um pouco menos fiel", 5, 0.90),
    ]
    assert choose_shortened(10, 4, options) == "curto e fiel"


def test_choose_shortened_returns_none_when_nothing_helps() -> None:
    assert choose_shortened(10, 4, [("igual", 10, 0.99), ("perdeu sentido", 5, 0.2)]) is None


def test_choose_shortened_never_cuts_below_the_floor() -> None:
    # Caso real: 11 sílabas escolhidas pra um alvo de 27 — o sentido foi junto.
    options = [("cortou demais", 11, 0.95), ("no alvo", 24, 0.90)]
    assert choose_shortened(45, 21, options) == "no alvo"


def test_has_forbidden_word_matches_whole_words_only() -> None:
    assert has_forbidden_word("Se tiver problema, manda msg.", PT_BR)
    assert has_forbidden_word("Aqui vcs.", PT_BR)
    assert not has_forbidden_word("Que mensagem bonita, quase.", PT_BR)


def test_judge_prompt_asks_yes_no_on_meaning_and_naturalness() -> None:
    prompt = build_judge_prompt("What do you mean?", "Quer dizer?", PT_BR)
    assert "What do you mean?" in prompt and "Quer dizer?" in prompt
    assert "Brazilian Portuguese" in prompt and "Answer only YES or NO" in prompt


def test_profanity_count_catches_inflections() -> None:
    # Caso real: o encurtamento trocou "Sua merda de gastos" por "Seus gastos".
    assert profanity_count("Sua merda de gastos, porra, que fodido.", PT_BR) == 3
    assert profanity_count("Seus gastos sozinhos no mês passado.", PT_BR) == 0


def test_shorten_prompt_shows_neighbors_and_keeps_sentence_edges() -> None:
    prompt = build_shorten_prompt(
        "instead of her growing up and getting a",
        "em vez de deixá-la crescer e conseguir um",
        8,
        PT_BR,
        previous_line="Por que estamos dando milhares pra ela",
        next_line="trabalho?",
    )
    assert '"trabalho?"' in prompt and "Por que estamos dando" in prompt
    assert "keep the words at that edge" in prompt


def test_shorten_prompt_without_neighbors_says_none() -> None:
    prompt = build_shorten_prompt("Hi.", "Oi, tudo bem com você?", 3, PT_BR)
    assert '- before: "(none)"' in prompt and '- after: "(none)"' in prompt


def test_sentence_groups_join_cut_sentences() -> None:
    texts = ["How much debt?", "Here you", "go. I gotta talk", "to this.", "Hi."]
    assert sentence_groups(texts) == [[0], [1, 2, 3], [4]]


def test_split_sentences() -> None:
    assert split_sentences("Hi. Who said that? Me!") == ["Hi.", "Who said that?", "Me!"]


def test_contracted_keeps_capital_and_verb_para() -> None:
    assert contracted("Está bom, você vai para casa.", PT_BR) == "Tá bom, cê vai pra casa."
    assert contracted("Para!", PT_BR) == "Para!"
    assert contracted("Eu estou indo para o carro.", PT_BR) == "Eu tô indo pro carro."


def test_choose_alternatives_keeps_best_when_it_fits() -> None:
    options = [[("uma frase longa", 0.0), ("curta", -0.1)]]
    assert choose_alternatives(options, 100, len) == ["uma frase longa"]


def test_choose_alternatives_shortens_only_within_score_drop() -> None:
    options = [
        [("aaaaaaaaaa", -0.1), ("aaaaa", -0.2)],
        [("bbbbbbbbbb", -0.1), ("b", -0.9)],  # curta, mas nota ruim demais
    ]
    assert choose_alternatives(options, 15, len, max_drop=0.25) == ["aaaaa", "bbbbbbbbbb"]


def test_split_by_source_follows_source_proportion() -> None:
    parts = split_by_source(
        "Aqui está. Preciso falar com isso.", ["Here you", "go. I gotta talk to this."]
    )
    assert len(parts) == 2 and all(parts)
    assert " ".join(parts) == "Aqui está. Preciso falar com isso."
    assert parts[0].startswith("Aqui")


def test_mt_options_moves_pronoun_and_drops_pt_pt() -> None:
    # Caso real do OPUS-MT: "mande-me", "tiveres" (pt-PT) no meio do n-best.
    options = mt_options(
        [
            ("Se tiveres um problema, manda-me uma mensagem.", -0.4),
            ("Se você tiver um problema, mande-me uma mensagem.", -0.5),
        ],
        PT_BR,
    )
    assert options[0] == ("Se você tiver um problema, me mande uma mensagem.", -0.5)
    assert all("tiveres" not in text for text, _ in options)
    assert ("Se cê tiver um problema, me mande uma mensagem.", -0.5) in options


def test_mt_options_keeps_capital_when_moving_pronoun() -> None:
    assert mt_options([("Mande-me o número.", -0.1)], PT_BR)[0][0] == "Me mande o número."


def test_mt_options_keeps_wrong_variant_when_nothing_else() -> None:
    assert mt_options([("Vais ver.", -0.1)], PT_BR)[0][0] == "Vais ver."


def test_mt_options_never_drops_profanity() -> None:
    options = mt_options(
        [("Seus gastos de merda foram milhares.", -0.5), ("Seus gastos foram milhares.", -0.6)],
        PT_BR,
    )
    assert all("merda" in text for text, _ in options)


def test_split_by_alignment_keeps_sentence_with_its_segment() -> None:
    # Caso real: "the quote is," | "if you have a problem" — a proporção
    # deixava "se cê" no fim da primeira fala e "tiver" na segunda.
    text = "a citação é, se cê tiver um problema"
    labels = [0, 0, 0, 1, 1, 1, 1, 1]
    assert split_by_alignment(text, labels, 2) == ["a citação é,", "se cê tiver um problema"]


def test_split_by_alignment_is_monotonic_against_noise() -> None:
    assert split_by_alignment("a b c d e f", [0, 1, 0, 0, 1, 1], 2) == ["a b c d", "e f"]


def test_split_by_alignment_gives_every_segment_a_word() -> None:
    assert split_by_alignment("Aqui tá. Tenho", [0, 0, 0], 3) == ["Aqui", "tá.", "Tenho"]
    assert split_by_alignment("Oi", [0], 2) is None


def test_sentence_groups_do_not_join_long_unpunctuated_lines() -> None:
    texts = [
        "everyone's nervous but you're not you're calm you're relaxed",
        "you're going to draw a lot of attention",
    ]
    assert sentence_groups(texts) == [[0], [1]]

