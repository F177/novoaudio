import pytest

from packages.pipeline.scorecard import (
    PROBLEMATIC_RATE_THRESHOLD,
    REVIEW_RATE_THRESHOLD,
    VERDICT_APROVADO,
    VERDICT_PROBLEMATICO,
    VERDICT_REVISAR,
    build_scorecard,
)


def _flags(**overrides: bool) -> dict[str, bool]:
    base = {
        "fora_duracao": False,
        "cer_alto": False,
        "truncado": False,
        "traducao_infiel": False,
        "clipping": False,
        "silencio_anormal": False,
        "locutor_suspeito": False,
        "entonacao_desalinhada": False,
        "emocao_incompativel": False,
        "cer_final_alto": False,
        "loop_confirmado": False,
        "segmento_curto": False,
    }
    base.update(overrides)
    return base


def _clean_segment(seg_id: str, cer: float = 0.05) -> dict:
    return {
        "id": seg_id,
        "t_inicio": 0.0,
        "t_fim": 1.0,
        "synthesized_ok": True,
        "evaluated_ok": True,
        "cer": cer,
        "cer_final": cer,
        "prosody_correlation": 0.8,
        "emotion_match": {"similarity": 0.9},
        "flags": _flags(),
        "any_flag": False,
    }


def test_build_scorecard_all_clean_is_aprovado() -> None:
    segments = [_clean_segment(f"seg{i:04d}") for i in range(10)]
    scorecard = build_scorecard(segments)
    assert scorecard["verdict"] == VERDICT_APROVADO
    assert scorecard["any_flag_rate"] == 0.0
    assert scorecard["total_segments"] == 10
    assert scorecard["synthesized_segments"] == 10


def test_build_scorecard_computes_flag_rates() -> None:
    segments = [_clean_segment(f"seg{i:04d}") for i in range(10)]
    for seg in segments[:3]:
        seg["flags"] = _flags(cer_alto=True)
        seg["any_flag"] = True
    scorecard = build_scorecard(segments)
    assert scorecard["flag_counts"]["cer_alto"] == 3
    assert scorecard["flag_rates"]["cer_alto"] == 0.3
    assert scorecard["any_flag_rate"] == 0.3


def test_build_scorecard_averages_objective_metrics() -> None:
    segments = [
        _clean_segment("seg0000", cer=0.1),
        _clean_segment("seg0001", cer=0.3),
    ]
    scorecard = build_scorecard(segments)
    assert scorecard["avg_cer"] == 0.2


def test_build_scorecard_review_verdict_at_threshold() -> None:
    total = 10
    n_flagged = int(REVIEW_RATE_THRESHOLD * total)
    segments = [_clean_segment(f"seg{i:04d}") for i in range(total)]
    for seg in segments[:n_flagged]:
        seg["flags"] = _flags(clipping=True)
        seg["any_flag"] = True
    scorecard = build_scorecard(segments)
    assert scorecard["verdict"] == VERDICT_REVISAR


def test_build_scorecard_problematico_verdict_above_threshold() -> None:
    total = 10
    n_flagged = int(PROBLEMATIC_RATE_THRESHOLD * total) + 1
    segments = [_clean_segment(f"seg{i:04d}") for i in range(total)]
    for seg in segments[:n_flagged]:
        seg["flags"] = _flags(silencio_anormal=True)
        seg["any_flag"] = True
    scorecard = build_scorecard(segments)
    assert scorecard["verdict"] == VERDICT_PROBLEMATICO


def test_build_scorecard_unsynthesized_segment_counts_as_any_flag_only() -> None:
    segments = [
        _clean_segment("seg0000"),
        {
            "id": "seg0001",
            "synthesized_ok": False,
            "evaluated_ok": False,
            "any_flag": True,
        },
    ]
    scorecard = build_scorecard(segments)
    assert scorecard["total_segments"] == 2
    assert scorecard["synthesized_segments"] == 1
    assert scorecard["any_flag_rate"] == 0.5
    # segmento sem síntese não deveria contaminar a média de CER
    assert scorecard["avg_cer"] == 0.05


def test_build_scorecard_empty_segments_does_not_crash() -> None:
    scorecard = build_scorecard([])
    assert scorecard["total_segments"] == 0
    assert scorecard["any_flag_rate"] == 0.0
    assert scorecard["avg_cer"] is None
    assert scorecard["verdict"] == VERDICT_APROVADO


def test_build_scorecard_worst_segments_sorted_by_flag_count_desc() -> None:
    segments = [_clean_segment(f"seg{i:04d}") for i in range(3)]
    segments[0]["flags"] = _flags(cer_alto=True)
    segments[0]["any_flag"] = True
    segments[1]["flags"] = _flags(cer_alto=True, clipping=True, truncado=True)
    segments[1]["any_flag"] = True
    scorecard = build_scorecard(segments, worst_n=5)
    worst_ids = [w["id"] for w in scorecard["worst_segments"]]
    assert worst_ids[0] == "seg0001"
    assert "seg0002" not in worst_ids


def test_build_scorecard_worst_segments_respects_worst_n() -> None:
    segments = [_clean_segment(f"seg{i:04d}") for i in range(5)]
    for seg in segments:
        seg["flags"] = _flags(cer_alto=True)
        seg["any_flag"] = True
    scorecard = build_scorecard(segments, worst_n=2)
    assert len(scorecard["worst_segments"]) == 2


def test_build_scorecard_missing_optional_metrics_are_none() -> None:
    segments = [
        {
            "id": "seg0000",
            "synthesized_ok": True,
            "evaluated_ok": False,
            "flags": _flags(),
            "any_flag": False,
        }
    ]
    scorecard = build_scorecard(segments)
    assert scorecard["avg_cer"] is None
    assert scorecard["avg_cer_final"] is None
    assert scorecard["avg_prosody_correlation"] is None
    assert scorecard["avg_emotion_similarity"] is None


def test_build_scorecard_averages_final_cer_independently_from_candidate_cer() -> None:
    segments = [
        _clean_segment("seg0000", cer=0.05),
        _clean_segment("seg0001", cer=0.05),
    ]
    # cer_final diverge do cer do candidato — simula uma corrupção que só
    # aparece no áudio FINAL (pós mixagem/transferência), não no candidato
    segments[0]["cer_final"] = 0.4
    segments[1]["cer_final"] = 0.5
    segments[0]["flags"]["cer_final_alto"] = True
    segments[0]["any_flag"] = True
    segments[1]["flags"]["cer_final_alto"] = True
    segments[1]["any_flag"] = True

    scorecard = build_scorecard(segments)
    assert scorecard["avg_cer"] == pytest.approx(0.05)
    assert scorecard["avg_cer_final"] == pytest.approx(0.45)
    assert scorecard["flag_rates"]["cer_final_alto"] == 1.0


def test_build_scorecard_flag_overlap_diagonal_matches_flag_counts() -> None:
    segments = [_clean_segment(f"seg{i:04d}") for i in range(3)]
    segments[0]["flags"] = _flags(cer_alto=True)
    segments[0]["any_flag"] = True
    scorecard = build_scorecard(segments)
    assert scorecard["flag_overlap"]["cer_alto"]["cer_alto"] == scorecard["flag_counts"]["cer_alto"]


def test_build_scorecard_flag_overlap_counts_segments_with_both_flags() -> None:
    segments = [_clean_segment(f"seg{i:04d}") for i in range(3)]
    # seg0000: só cer_alto. seg0001: cer_alto E clipping juntos. seg0002: limpo.
    segments[0]["flags"] = _flags(cer_alto=True)
    segments[0]["any_flag"] = True
    segments[1]["flags"] = _flags(cer_alto=True, clipping=True)
    segments[1]["any_flag"] = True

    scorecard = build_scorecard(segments)
    overlap = scorecard["flag_overlap"]
    assert overlap["cer_alto"]["cer_alto"] == 2  # seg0000 + seg0001
    assert overlap["clipping"]["clipping"] == 1  # só seg0001
    assert overlap["cer_alto"]["clipping"] == 1  # só seg0001 tem as duas
    assert overlap["clipping"]["cer_alto"] == 1  # simétrico


def test_build_scorecard_flag_overlap_is_zero_when_no_segment_has_both() -> None:
    segments = [_clean_segment(f"seg{i:04d}") for i in range(2)]
    segments[0]["flags"] = _flags(cer_alto=True)
    segments[0]["any_flag"] = True
    segments[1]["flags"] = _flags(clipping=True)
    segments[1]["any_flag"] = True

    scorecard = build_scorecard(segments)
    overlap = scorecard["flag_overlap"]
    assert overlap["cer_alto"]["clipping"] == 0
    assert overlap["clipping"]["cer_alto"] == 0


def test_build_scorecard_segments_with_multiple_flags_counts_two_or_more() -> None:
    segments = [_clean_segment(f"seg{i:04d}") for i in range(3)]
    segments[0]["flags"] = _flags(cer_alto=True)  # só 1 flag
    segments[0]["any_flag"] = True
    segments[1]["flags"] = _flags(cer_alto=True, clipping=True)  # 2 flags
    segments[1]["any_flag"] = True
    segments[2]["flags"] = _flags(cer_alto=True, clipping=True, truncado=True)  # 3 flags
    segments[2]["any_flag"] = True

    scorecard = build_scorecard(segments)
    assert scorecard["segments_with_multiple_flags"] == 2  # seg0001 e seg0002


def test_build_scorecard_segments_with_multiple_flags_zero_when_none_overlap() -> None:
    segments = [_clean_segment(f"seg{i:04d}") for i in range(2)]
    segments[0]["flags"] = _flags(cer_alto=True)
    segments[0]["any_flag"] = True
    scorecard = build_scorecard(segments)
    assert scorecard["segments_with_multiple_flags"] == 0
