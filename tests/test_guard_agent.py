import importlib.util
import json
import subprocess
import sys
from pathlib import Path

HOOK = Path(__file__).resolve().parents[1] / ".claude" / "hooks" / "guard_agent.py"
_spec = importlib.util.spec_from_file_location("guard_agent", HOOK)
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)


def _root(tmp_path: Path) -> Path:
    golden = tmp_path / "evals" / "golden"
    golden.mkdir(parents=True)
    shorts = [{"video": "devclip", "split": "dev"}, {"video": "segredo", "split": "holdout"}]
    (golden / "en-xx.json").write_text(json.dumps({"shorts": shorts}), encoding="utf-8")
    return tmp_path


def test_nobody_touches_gpu_or_pushes(tmp_path: Path) -> None:
    root = _root(tmp_path)
    for role in ("judge", "analyst", "implementer"):
        for cmd in ("python -m scripts.dub videos/a.mp4", "ssh root@pod", "git push origin x"):
            assert guard.check(role, "Bash", {"command": cmd}, root)
    allowed = {"command": "python -m scripts.judge_audio x"}
    assert guard.check("judge", "Bash", allowed, root) is None


def test_judge_and_analyst_write_only_in_cache(tmp_path: Path) -> None:
    root = _root(tmp_path)
    assert guard.check("judge", "Write", {"file_path": str(root / "packages/x.py")}, root)
    ok = {"file_path": str(root / ".cache/cycles/c/base/labels/dev/devclip.jsonl")}
    assert guard.check("judge", "Write", ok, root) is None


def test_analyst_sees_dev_but_never_holdout(tmp_path: Path) -> None:
    root = _root(tmp_path)
    dev = {"file_path": ".cache/cycles/c/base/labels/dev/devclip.jsonl"}
    assert guard.check("analyst", "Read", dev, root) is None
    for blocked in ({"file_path": ".cache/cycles/c/base/labels/holdout/segredo.jsonl"},
                    {"command": "cat .cache/dub/segredo/report.json"},
                    {"pattern": "seu pai", "path": ".cache"},
                    {"command": "grep -r pai .cache/cycles/c/base"}):
        tool = "Bash" if "command" in blocked else "Read"
        assert guard.check("analyst", tool, blocked, root), blocked
    snap = {"command": "python -m scripts.judge_packet .cache/cycles/c/base/dub/devclip"}
    assert guard.check("analyst", "Bash", snap, root) is None


def test_implementer_never_sees_eval_set_or_protected_files(tmp_path: Path) -> None:
    root = _root(tmp_path)
    assert guard.check("implementer", "Read", {"file_path": ".cache/dub/devclip/report.json"},
                       root)
    for path in ("evals/golden/en-xx.json", ".claude/agents/dub-judge.md", "TASKS.md"):
        assert guard.check("implementer", "Edit", {"file_path": str(root / path)}, root)
    code = {"file_path": str(root / "packages/pipeline/translation.py")}
    assert guard.check("implementer", "Edit", code, root) is None


def test_hook_exits_2_with_reason(tmp_path: Path) -> None:
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": "ssh root@pod"}})
    proc = subprocess.run([sys.executable, str(HOOK), "analyst"], input=payload, text=True,
                          capture_output=True, env={"CLAUDE_PROJECT_DIR": str(_root(tmp_path))})
    assert proc.returncode == 2 and "GPU/Pod" in proc.stderr


def _root_with_exam(tmp_path: Path) -> Path:
    root = _root(tmp_path)
    golden = root / "evals" / "golden"
    data = json.loads((golden / "en-xx.json").read_text())
    data["shorts"].append({"video": "selado", "split": "exame"})
    (golden / "en-xx.json").write_text(json.dumps(data), encoding="utf-8")
    return root


def test_new_roles_and_rules(tmp_path: Path) -> None:
    root = _root_with_exam(tmp_path)
    # ninguém sobe o servidor de escuta
    for role in ("judge", "analyst", "implementer", "reviewer"):
        assert guard.check(role, "Bash", {"command": "python -m scripts.listen c base"}, root)
    # juiz não vê o mapa do A/B cego
    assert guard.check("judge", "Read", {"file_path": ".cache/cycles/c/blind/mapping.json"}, root)
    blind_side = {"command": "python -m scripts.judge_packet .cache/cycles/c/blind/x/A"}
    assert guard.check("judge", "Bash", blind_side, root) is None
    # analista: exame bloqueado; rótulos humanos só de dev; sondas liberadas
    assert guard.check("analyst", "Bash", {"command": "cat .cache/dub/selado/report.json"}, root)
    human_dev = {"file_path": "evals/human_labels/en-xx/devclip.jsonl"}
    assert guard.check("analyst", "Read", human_dev, root) is None
    for blocked in ("evals/human_labels/en-xx/segredo.jsonl", "evals/human_labels/en-xx"):
        assert guard.check("analyst", "Read", {"file_path": blocked}, root)
    probe = {"command": "python .cache/cycles/c/sondas/rerank.py"}
    assert guard.check("analyst", "Bash", probe, root) is None
    # revisor: não vê o conjunto, só escreve em tests/
    assert guard.check("reviewer", "Read", {"file_path": "evals/human_labels/en-xx/devclip.jsonl"},
                       root)
    assert guard.check("reviewer", "Write", {"file_path": str(root / "packages/x.py")}, root)
    ok_test = {"file_path": str(root / "tests/test_adversarial_t1.py")}
    assert guard.check("reviewer", "Write", ok_test, root) is None
    # implementador também não lê rótulos humanos
    assert guard.check("implementer", "Bash", {"command": "cat evals/human_labels/x/a.jsonl"}, root)
