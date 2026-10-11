import json
from pathlib import Path

from scripts.eval_blind import prepare, unblind
from scripts.eval_common import labels_path, read_labels, write_labels
from scripts.eval_cycle import carry
from scripts.judge_packet import load_run
from tests.cycle_fixtures import judge_labels, make_cycle, make_run


def test_blind_round_trip(tmp_path: Path) -> None:
    root, cid = make_cycle(tmp_path, [("a", "dev", "x"), ("z", "dev", "y")])
    cdir = root / cid
    make_run(cdir / "base" / "dub" / "a", ["um", "ruim", "tres"])
    make_run(cdir / "cand" / "dub" / "a", ["um", "bom", "tres"])
    make_run(cdir / "base" / "dub" / "z", ["igual"])
    make_run(cdir / "cand" / "dub" / "z", ["igual"])
    judge_labels(cdir, "base", "dev", "a", {"seg0001": "sentido.literal"})
    judge_labels(cdir, "base", "dev", "z", {})
    carry(cid, "base", "cand", root)

    counts = prepare(cid, "base", "cand", root, seed=3)
    assert counts == {"shorts": 1, "segmentos": 1}  # z não mudou, não entra no cego
    mapping = json.loads((cdir / "blind" / "mapping.json").read_text())
    assert set(mapping["a"].values()) >= {"base", "cand"} and "z" not in mapping
    for letter in "AB":
        side = cdir / "blind" / "a" / letter
        assert (side / "pending_A.txt").exists() is False
        assert [p.name for p in (side / "final_segments").iterdir()] == ["seg0001.wav"]
        assert (cdir / "blind" / "a" / f"pending_{letter}.txt").read_text().split() == ["seg0001"]

    # o juiz rotula às cegas: a versão "boa" (cand) sem problema
    cand_letter = "A" if mapping["a"]["A"] == "cand" else "B"
    base_letter = "B" if cand_letter == "A" else "A"
    write_labels(cdir / "blind" / "a" / f"labels_{cand_letter}.jsonl",
                 [{"segment_id": "seg0001", "rev": "xx", "rotulo": "ok", "classe": None}])
    write_labels(cdir / "blind" / "a" / f"labels_{base_letter}.jsonl",
                 [{"segment_id": "seg0001", "rev": "xx", "rotulo": "corrigiria",
                   "classe": "sentido.literal"}])
    (cdir / "blind" / "a" / "preferencias.jsonl").write_text(
        json.dumps({"A": ["seg0001"], "B": ["seg0001"], "melhor": cand_letter}) + "\n")

    result = unblind(cid, "base", "cand", root)
    assert result["preferencias"] == {"base": 0, "candidato": 1, "igual": 0}
    cand_rows, _ = read_labels(labels_path(cdir, "cand", "dev", "a"))
    by_id = {r["segment_id"]: r for r in cand_rows}
    real_rev = load_run(cdir / "cand" / "dub" / "a")[1][1].rev
    assert by_id["seg0001"]["rev"] == real_rev and by_id["seg0001"]["cego"] is True
    assert by_id["seg0000"].get("herdado_de") == "base"
    paired, _ = read_labels(labels_path(cdir, "base-par", "dev", "a"))
    assert {r["segment_id"]: r["rotulo"] for r in paired}["seg0001"] == "corrigiria"
    assert read_labels(labels_path(cdir, "base-par", "dev", "z"))[0]  # cópia da base
