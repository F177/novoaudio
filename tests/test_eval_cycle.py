import json
from pathlib import Path

from scripts.eval_cycle import (
    carry,
    compare,
    labels_path,
    new_cycle,
    read_labels,
    snapshot,
    summarize,
    write_labels,
)
from scripts.judge_packet import segment_revision

TAXONOMY = Path(__file__).resolve().parents[1] / "evals" / "taxonomy.json"


def _cycle(tmp_path: Path, shorts: list[tuple[str, str, str]]) -> tuple[Path, str]:
    manifest = {
        "shorts": [{"video": v, "split": sp, "categoria": c} for v, sp, c in shorts],
        "elegibilidade": {"min_shorts": 2, "min_categorias": 2},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    root = tmp_path / "cycles"
    return root, new_cycle(path, "t", root)


def _run(directory: Path, texts: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    segs = [{"id": f"seg{i:04d}", "t_inicio": float(i), "t_fim": i + 1.0,
             "texto_original": "x", "traducao": t} for i, t in enumerate(texts)]
    (directory / "report.json").write_text(json.dumps({"segments": segs}), encoding="utf-8")


def _label(root: Path, cid: str, label: str, split: str, video: str,
           problems: dict[str, str], n_segments: int = 3, short: str = "publicaria") -> None:
    rows = [{"segment_id": f"seg{i:04d}", "rotulo": "corrigiria" if f"seg{i:04d}" in problems
             else "ok", "classe": problems.get(f"seg{i:04d}")} for i in range(n_segments)]
    write_labels(labels_path(root / cid, label, split, video), [*rows, {"short": short}])


def test_snapshot_copies_runs_and_reports_missing(tmp_path: Path) -> None:
    root, cid = _cycle(tmp_path, [("a", "dev", "x"), ("b", "dev", "x")])
    dub = tmp_path / "dub"
    _run(dub / "a", ["Oi."])
    (dub / "a" / "final_segments").mkdir()
    (dub / "a" / "final_segments" / "seg0000.wav").write_bytes(b"RIFF")
    assert snapshot(cid, "base", root, dub) == ["b"]
    assert (root / cid / "base" / "dub" / "a" / "final_segments" / "seg0000.wav").exists()


def test_carry_reuses_identical_segments_and_lists_changed_ones(tmp_path: Path) -> None:
    root, cid = _cycle(tmp_path, [("a", "dev", "x")])
    _run(root / cid / "candidato" / "dub" / "a", ["Oi.", "Liga pro seu pai."])
    old = [{"segment_id": "seg0000", "rev": segment_revision("Oi."), "rotulo": "ok"},
           {"segment_id": "seg0001", "rev": segment_revision("Chame seu pai."),
            "rotulo": "corrigiria", "classe": "sentido.literal"},
           {"short": "nao_publicaria"}]
    write_labels(labels_path(root / cid, "base", "dev", "a"), old)
    assert carry(cid, "base", "candidato", root) == {"herdados": 1, "pendentes": 1}
    carried, short = read_labels(labels_path(root / cid, "candidato", "dev", "a"))
    assert [r["segment_id"] for r in carried] == ["seg0000"] and short is None
    pending = (root / cid / "candidato" / "pending" / "a.txt").read_text()
    assert pending.split() == ["seg0001"]


def test_summary_eligibility_needs_spread_across_shorts_and_categories(tmp_path: Path) -> None:
    root, cid = _cycle(tmp_path, [("a", "dev", "streamer"), ("b", "dev", "podcast"),
                                  ("c", "dev", "podcast")])
    _label(root, cid, "base", "dev", "a", {"seg0000": "sentido.pessoa_sujeito"})
    _label(root, cid, "base", "dev", "b", {"seg0001": "sentido.pessoa_sujeito"})
    _label(root, cid, "base", "dev", "c", {"seg0000": "voz.instavel", "seg0002": "voz.instavel"})
    s = summarize(cid, "base", "dev", root, TAXONOMY)
    assert s["classes"]["sentido.pessoa_sujeito"]["elegivel"]
    assert not s["classes"]["voz.instavel"]["elegivel"]  # 1 Short só, mesmo com 2 casos


def _ab(tmp_path: Path, dev: tuple[int, int], hold: tuple[int, int],
        other_dev: tuple[int, int] = (0, 0)) -> str:
    root, cid = _cycle(tmp_path, [("a", "dev", "x"), ("h", "holdout", "y")])
    target = "sentido.pessoa_sujeito"
    for label, i in (("base", 0), ("cand", 1)):
        problems = {f"seg{k:04d}": target for k in range(dev[i])}
        problems |= {f"seg{k:04d}": "registro.formal_demais"
                     for k in range(dev[i], dev[i] + other_dev[i])}
        _label(root, cid, label, "dev", "a", problems, n_segments=8)
        _label(root, cid, label, "holdout", "h",
               {f"seg{k:04d}": target for k in range(hold[i])}, n_segments=8)
    return compare(cid, "base", "cand", [target], root, TAXONOMY)["veredito"]


def test_compare_accepts_real_improvement(tmp_path: Path) -> None:
    assert _ab(tmp_path, dev=(4, 1), hold=(2, 1)).startswith("ACEITAR, sujeito")


def test_compare_rejects_when_holdout_gets_worse(tmp_path: Path) -> None:
    assert "overfitting" in _ab(tmp_path, dev=(4, 0), hold=(1, 2))


def test_compare_rejects_regression_in_other_class(tmp_path: Path) -> None:
    assert "regressões" in _ab(tmp_path, dev=(4, 1), hold=(1, 1), other_dev=(0, 3))


def test_compare_inconclusive_without_dev_gain(tmp_path: Path) -> None:
    assert _ab(tmp_path, dev=(3, 3), hold=(1, 1)).startswith("INCONCLUSIVO")


def test_compare_flags_unverified_generalization(tmp_path: Path) -> None:
    assert _ab(tmp_path, dev=(3, 1), hold=(0, 0)).startswith("ACEITAR COM RESSALVA")
