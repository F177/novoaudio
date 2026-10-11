

from packages.pipeline.translation import (
    PT_BR,
    choose_alternatives,
    contracted,
    mt_options,
    profanity_count,
    rank_candidates,
    score_candidate,
    sentence_groups,
    split_by_alignment,
    split_by_source,
    split_sentences,
)


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


def test_profanity_count_catches_inflections() -> None:
    # Caso real: o encurtamento trocou "Sua merda de gastos" por "Seus gastos".
    assert profanity_count("Sua merda de gastos, porra, que fodido.", PT_BR) == 3
    assert profanity_count("Seus gastos sozinhos no mês passado.", PT_BR) == 0


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
