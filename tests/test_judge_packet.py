import json
from pathlib import Path

from scripts.judge_packet import (
    ReviewSegment,
    active_flags,
    load_run,
    pair_segments,
    render_pair,
    text_revision,
)


def _seg(id_: str, start: float, end: float, pt: str) -> ReviewSegment:
    return ReviewSegment(id=id_, start=start, end=end, source_text="x", dub_text=pt)


def test_text_revision_ignores_whitespace_but_not_words() -> None:
    assert text_revision("Liga pro  seu pai.") == text_revision(" Liga pro seu pai.")
    assert text_revision("Liga pro seu pai.") != text_revision("Chame seu pai.")


def test_active_flags_skips_false_and_stub_markers() -> None:
    flags = {"cer_alto": True, "truncado": False, "traducao_infiel_not_computed": True}
    assert active_flags(flags) == ("cer_alto",)


def test_load_traducao_json(tmp_path: Path) -> None:
    path = tmp_path / "short__2_so-opus_traducao.json"
    item = {"id": "seg0000", "t_inicio": 0.2, "t_fim": 1.6,
            "texto_original": "Call your dad.", "traducao_usada": "Chame seu pai."}
    path.write_text(json.dumps([item]), encoding="utf-8")
    name, segs = load_run(path)
    assert name == "short__2_so-opus"
    assert segs[0].dub_text == "Chame seu pai."
    assert segs[0].asr_text is None


def test_load_cache_dir_merges_speaker_and_flags(tmp_path: Path) -> None:
    report = {"segments": [{
        "id": "seg0000", "t_inicio": 0.0, "t_fim": 2.0, "texto_original": "Hi",
        "traducao": "Oi", "hipotese_final": "oi", "cer_final": 0.0,
        "target_seconds": 2.0, "achieved_seconds": 1.9,
        "flags": {"fora_duracao": True, "locutor_suspeito_not_computed": True},
    }]}
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")
    speakers = [{"id": "seg0000", "speaker": "SPEAKER_01"}]
    (tmp_path / "segments.json").write_text(json.dumps(speakers), encoding="utf-8")
    _, segs = load_run(tmp_path)
    assert segs[0].speaker == "SPEAKER_01"
    assert segs[0].flags == ("fora_duracao",)
    assert segs[0].asr_text == "oi"


def test_pair_segments_survives_resegmentation() -> None:
    a = [_seg("seg0000", 0.0, 4.0, "Oi. Tudo bem?")]
    b = [_seg("seg0000", 0.0, 2.0, "Oi."), _seg("seg0001", 2.0, 4.0, "Tudo bem?"),
         _seg("seg0002", 9.0, 10.0, "Tchau.")]
    pairs, unmatched = pair_segments(a, b)
    assert [s.id for s in pairs[0][1]] == ["seg0000", "seg0001"]
    assert [s.id for s in unmatched] == ["seg0002"]


def test_render_pair_marks_identical_and_counts_changes() -> None:
    a = [_seg("seg0000", 0.0, 1.0, "Oi."), _seg("seg0001", 1.0, 2.0, "Chame seu pai.")]
    b = [_seg("seg0000", 0.0, 1.0, "Oi."), _seg("seg0001", 1.0, 2.0, "Liga pro seu pai.")]
    out = render_pair("A", a, "B", b)
    assert "1 de 2 segmentos de A mudaram" in out
    assert "(igual) Oi." in out
    assert "B : Liga pro seu pai." in out
