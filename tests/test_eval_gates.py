"""Portões do `compare`: ruído, teste do sinal, custo, rótulo humano, calibração, preferência."""

import json
from pathlib import Path

from scripts.eval_common import add_human, judge_model, read_jsonl, write_json
from scripts.eval_cycle import carry, compare, holdout_uses, summarize
from scripts.judge_packet import load_run
from tests.cycle_fixtures import judge_labels, make_cycle, make_run

TAXONOMY = Path(__file__).resolve().parents[1] / "evals" / "taxonomy.json"
T = "sentido.pessoa_sujeito"
OTHER = "registro.formal_demais"


def _cmp(root: Path, cid: str, tmp_path: Path, **kw) -> dict:
    return compare(cid, "base", "cand", [T], root, TAXONOMY, human_root=tmp_path / "human",
                   calib_root=tmp_path / "calib", **kw)


def _two_runs(root: Path, cid: str, video: str, base: list[str], cand: list[str],
              cost: tuple[float, float] | None = None) -> None:
    cdir = root / cid
    make_run(cdir / "base" / "dub" / video, base, {"synthesize": cost[0]} if cost else None)
    make_run(cdir / "cand" / "dub" / video, cand, {"synthesize": cost[1]} if cost else None)


def test_improvement_inside_judge_noise_is_inconclusive(tmp_path: Path) -> None:
    root, cid = make_cycle(tmp_path, [("a", "dev", "x"), ("h", "holdout", "y")])
    for v, sp in (("a", "dev"), ("h", "holdout")):
        _two_runs(root, cid, v, ["p"] * 8, ["p"] * 8)
        judge_labels(root / cid, "base", sp, v, {"seg0000": T, "seg0001": T} if v == "a" else {})
        judge_labels(root / cid, "cand", sp, v, {"seg0000": T} if v == "a" else {})
    # 8 segmentos reavaliados × ruído padrão 0,15 = 1,2 > melhora de 1
    assert _cmp(root, cid, tmp_path)["veredito"].startswith("INCONCLUSIVO: a melhora")


def test_improvement_concentrated_in_one_short_fails_sign_test(tmp_path: Path) -> None:
    root, cid = make_cycle(tmp_path, [("a", "dev", "x"), ("b", "dev", "y"), ("c", "dev", "z"),
                                      ("h", "holdout", "y")])
    cdir = root / cid
    plan = {"a": ({f"seg{i:04d}": T for i in range(4)}, {}),
            "b": ({}, {"seg0000": T}), "c": ({}, {"seg0000": T}), "h": ({"seg0000": T}, {})}
    for v, (base_p, cand_p) in plan.items():
        sp = "holdout" if v == "h" else "dev"
        _two_runs(root, cid, v, ["p"] * 6, ["p"] * 6)
        judge_labels(cdir, "base", sp, v, base_p)
        judge_labels(cdir, "cand", sp, v, cand_p, rejudged=set(base_p) | set(cand_p))
    r = _cmp(root, cid, tmp_path)
    assert r["splits"]["dev"]["sinal"] == {"melhoraram": 1, "pioraram": 2, "p": 1.0}
    assert "mais Shorts" in r["veredito"]


def test_cost_increase_is_a_regression(tmp_path: Path) -> None:
    root, cid = make_cycle(tmp_path, [("a", "dev", "x"), ("h", "holdout", "y")])
    for v, sp in (("a", "dev"), ("h", "holdout")):
        _two_runs(root, cid, v, ["p"] * 4, ["p"] * 4, cost=(300.0, 420.0))
        judge_labels(root / cid, "base", sp, v, {f"seg{i:04d}": T for i in range(4)})
        judge_labels(root / cid, "cand", sp, v, {}, rejudged=set())
    r = _cmp(root, cid, tmp_path)
    assert any("custo por Short 300 s → 420 s" in x for x in r["regressoes"])
    assert r["veredito"].startswith("REJEITAR")


def test_human_label_replaces_judge_label(tmp_path: Path) -> None:
    root, cid = make_cycle(tmp_path, [("a", "dev", "x")])
    make_run(root / cid / "base" / "dub" / "a", ["p", "q"])
    judge_labels(root / cid, "base", "dev", "a", {"seg0001": T})
    rev = load_run(root / cid / "base" / "dub" / "a")[1][1].rev
    human = tmp_path / "human"
    assert summarize(cid, "base", "dev", root, TAXONOMY, human)["classes"][T]["corrigiria"] == 1
    add_human("en-xx", {"video": "a", "segment_id": "seg0001", "rev": rev, "rotulo": "ok",
                        "classe": None}, human)
    s = summarize(cid, "base", "dev", root, TAXONOMY, human)
    assert T not in s["classes"] and s["fontes"]["humano"] == 1


def test_uncalibrated_target_needs_human_labels_on_changed_segments(tmp_path: Path) -> None:
    root, cid = make_cycle(tmp_path, [("a", "dev", "x"), ("b", "dev", "y"), ("h", "holdout", "y")])
    cdir = root / cid
    for v in ("a", "b", "h"):
        sp = "holdout" if v == "h" else "dev"
        make_run(cdir / "base" / "dub" / v, ["ruim", "ok1", "ok2", "ok3"])
        make_run(cdir / "cand" / "dub" / v, ["bom", "ok1", "ok2", "ok3"])
        judge_labels(cdir, "base", sp, v, {"seg0000": T})
    carry(cid, "base", "cand", root)
    for v in ("a", "b", "h"):
        judge_labels(cdir, "cand", "holdout" if v == "h" else "dev", v, {}, rejudged={"seg0000"})
    write_json(tmp_path / "calib" / "en-xx.json", {
        "juiz_modelo": judge_model(), "ruido": {"discordancia": 0.05, "n": 50},
        "classes": {T: {"confiavel": False}}})
    r = _cmp(root, cid, tmp_path)
    assert r["veredito"].startswith("PRECISA DE RÓTULO HUMANO")
    for v in ("a", "b", "h"):
        seg = load_run(cdir / "cand" / "dub" / v)[1][0]
        add_human("en-xx", {"video": v, "segment_id": seg.id, "rev": seg.rev, "rotulo": "ok",
                            "classe": None}, tmp_path / "human")
    assert _cmp(root, cid, tmp_path)["veredito"].startswith("ACEITAR, sujeito")


def test_blind_judge_preferring_base_is_a_regression(tmp_path: Path) -> None:
    root, cid = make_cycle(tmp_path, [("a", "dev", "x"), ("h", "holdout", "y")])
    for v, sp in (("a", "dev"), ("h", "holdout")):
        _two_runs(root, cid, v, ["p"] * 4, ["p"] * 4)
        judge_labels(root / cid, "base", sp, v, {f"seg{i:04d}": T for i in range(4)})
        judge_labels(root / cid, "cand", sp, v, {}, rejudged=set())
    write_json(root / cid / "preferencias_juiz.json", {"base": 3, "candidato": 1, "igual": 0})
    assert any("preferiu a base" in x for x in _cmp(root, cid, tmp_path)["regressoes"])


def test_compare_records_holdout_use(tmp_path: Path) -> None:
    root, cid = make_cycle(tmp_path, [("a", "dev", "x"), ("h", "holdout", "y")])
    for v, sp in (("a", "dev"), ("h", "holdout")):
        _two_runs(root, cid, v, ["p"], ["p"])
        judge_labels(root / cid, "base", sp, v, {})
        judge_labels(root / cid, "cand", sp, v, {})
    _cmp(root, cid, tmp_path)
    meta = json.loads((root / cid / "cycle.json").read_text())
    assert holdout_uses(meta["manifest"], root) == 1
    assert read_jsonl(root / cid / "nao-existe.jsonl") == []


def test_exam_split_stays_out_of_regular_snapshots(tmp_path: Path) -> None:
    from scripts.eval_cycle import snapshot

    root, cid = make_cycle(tmp_path, [("a", "dev", "x"), ("e", "exame", "y")])
    dub = tmp_path / "dub"
    make_run(dub / "a", ["p"])
    make_run(dub / "e", ["p"])
    assert snapshot(cid, "base", root, dub) == []
    assert not (root / cid / "base" / "dub" / "e").exists()
    assert snapshot(cid, "marco", root, dub, splits=("exame",)) == []
    assert (root / cid / "marco" / "dub" / "e").exists()
