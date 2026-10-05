from packages.pipeline.diagnostics import DIAGNOSTICS, build_diagnostics


def _scorecard(flag_rates: dict[str, float]) -> dict:
    return {"flag_rates": flag_rates}


def test_build_diagnostics_includes_flags_above_threshold() -> None:
    scorecard = _scorecard({"entonacao_desalinhada": 0.5, "clipping": 0.0})
    diagnostics = build_diagnostics(scorecard, threshold=0.2)
    flags = [d["flag"] for d in diagnostics]
    assert "entonacao_desalinhada" in flags
    assert "clipping" not in flags


def test_build_diagnostics_excludes_flags_below_threshold() -> None:
    scorecard = _scorecard({"cer_alto": 0.1})
    diagnostics = build_diagnostics(scorecard, threshold=0.2)
    assert diagnostics == []


def test_build_diagnostics_sorted_by_rate_descending() -> None:
    scorecard = _scorecard({"entonacao_desalinhada": 0.3, "emocao_incompativel": 0.8})
    diagnostics = build_diagnostics(scorecard, threshold=0.2)
    assert [d["flag"] for d in diagnostics] == ["emocao_incompativel", "entonacao_desalinhada"]


def test_build_diagnostics_known_flags_have_real_content() -> None:
    scorecard = _scorecard({"entonacao_desalinhada": 1.0})
    diagnostics = build_diagnostics(scorecard, threshold=0.2)
    entry = diagnostics[0]
    assert entry["titulo"]
    assert len(entry["causa_raiz"]) > 20
    assert len(entry["como_resolver"]) > 0
    assert entry["kpi_impacto"]


def test_build_diagnostics_unknown_flag_still_reported_without_crashing() -> None:
    scorecard = _scorecard({"traducao_infiel": 0.9})
    diagnostics = build_diagnostics(scorecard, threshold=0.2)
    assert diagnostics[0]["flag"] == "traducao_infiel"
    assert diagnostics[0]["como_resolver"] == []


def test_build_diagnostics_respects_default_threshold_constant() -> None:
    # usa o threshold default (REVIEW_RATE_THRESHOLD) sem passar explícito
    scorecard = _scorecard({"cer_final_alto": 0.99})
    diagnostics = build_diagnostics(scorecard)
    assert len(diagnostics) == 1


def test_diagnostics_knowledge_base_entries_are_well_formed() -> None:
    for name, entry in DIAGNOSTICS.items():
        assert entry["titulo"], name
        assert entry["causa_raiz"], name
        assert isinstance(entry["como_resolver"], list), name
        assert entry["kpi_impacto"], name
