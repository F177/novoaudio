from pathlib import Path

from scripts.eval_calibration import (
    agreement,
    calibrate,
    class_table,
    load_calibration,
    noise_rate,
    trusted_classes,
)
from scripts.eval_common import add_human, judge_model, labels_path, write_labels
from scripts.judge_packet import load_run
from tests.cycle_fixtures import judge_labels, make_cycle, make_run

RULE = {"min_n": 5, "min_precisao": 0.7, "min_recall": 0.6}


def _row(rotulo: str, classe: str | None = None) -> dict:
    return {"rotulo": rotulo, "classe": classe}


def test_agreement_per_class_and_trust() -> None:
    pairs = [(_row("corrigiria", "a"), _row("corrigiria", "a"))] * 5
    pairs += [(_row("corrigiria", "a"), _row("ok"))]               # alarme falso
    pairs += [(_row("ok"), _row("corrigiria", "a"))]               # o juiz não viu
    pairs += [(_row("escutar", "b"), _row("corrigiria", "b"))] * 2  # pouca amostra
    pairs += [(_row("ok"), _row("ok"))] * 10
    agr = agreement(pairs)
    table = class_table(agr, RULE)
    assert table["a"]["precisao"] == round(5 / 6, 2) and table["a"]["recall"] == round(5 / 6, 2)
    assert table["a"]["confiavel"] and not table["b"]["confiavel"]
    assert agr["vp"] == 7 and agr["fp"] == 1 and agr["fn"] == 1 and agr["vn"] == 10


def test_judge_noise_counts_class_flips() -> None:
    pairs = [(_row("corrigiria", "a"), _row("corrigiria", "b")), (_row("ok"), _row("ok"))]
    assert agreement(pairs)["discordancia"] == 0.5


def test_calibrate_writes_file_valid_only_for_same_judge_model(tmp_path: Path) -> None:
    root, cid = make_cycle(tmp_path, [("a", "dev", "x")])
    cdir = root / cid
    make_run(cdir / "base" / "dub" / "a", [f"t{i}" for i in range(6)])
    problems = {f"seg{i:04d}": "sentido.literal" for i in range(5)}
    judge_labels(cdir, "base", "dev", "a", problems)
    rows = [{"segment_id": s.id, "rev": s.rev, "rotulo": "corrigiria" if s.id in problems
             else "ok", "classe": problems.get(s.id)}
            for s in load_run(cdir / "base" / "dub" / "a")[1]]
    rows[0] = {**rows[0], "classe": "sentido.omissao"}  # segunda leitura troca uma classe
    write_labels(labels_path(cdir, "base-r2", "dev", "a"), rows)
    for s in load_run(cdir / "base" / "dub" / "a")[1]:
        add_human("en-xx", {"video": "a", "segment_id": s.id, "rev": s.rev,
                            "rotulo": "corrigiria" if s.id in problems else "ok",
                            "classe": problems.get(s.id)}, tmp_path / "human")
    result = calibrate(cid, "base", "base-r2", root, tmp_path / "human", tmp_path / "calib")
    assert result["classes"]["sentido.literal"]["confiavel"]
    assert result["ruido"]["discordancia"] == round(1 / 6, 3)
    calib = load_calibration("en-xx", judge_model(), tmp_path / "calib")
    assert trusted_classes(calib) == {"sentido.literal"}
    assert noise_rate(calib) == (round(1 / 6, 3), True)
    other = load_calibration("en-xx", "outro-modelo", tmp_path / "calib")
    assert trusted_classes(other) == set() and noise_rate(other)[1] is False
