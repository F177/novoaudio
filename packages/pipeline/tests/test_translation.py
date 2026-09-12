import json

import pytest

from packages.pipeline.translation import (
    build_prompt,
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


def test_score_candidate_prefers_matching_budget() -> None:
    exact = score_candidate("Uma frase com doze silabas mais ou menos aqui", target_syllables=12)
    way_off = score_candidate("Oi", target_syllables=12)
    assert exact.budget_score > way_off.budget_score


def test_score_candidate_detects_spoken_markers() -> None:
    natural = score_candidate("Cê tá indo pra casa?", target_syllables=6)
    formal = score_candidate("Você está indo para a residência?", target_syllables=8)
    assert natural.naturalness_score > formal.naturalness_score


def test_rank_candidates_orders_best_first() -> None:
    candidates = [
        "Isso e uma frase bem mais longa do que o orcamento pedia originalmente",
        "Isso cabe certinho",
        "Oi",
    ]
    ranked = rank_candidates(candidates, target_syllables=6)
    assert ranked[0].text == "Isso cabe certinho"
    assert ranked == sorted(ranked, key=lambda c: c.score, reverse=True)


def test_rank_candidates_returns_one_result_per_candidate() -> None:
    candidates = ["A.", "B.", "C.", "D.", "E."]
    ranked = rank_candidates(candidates, target_syllables=1)
    assert len(ranked) == 5
