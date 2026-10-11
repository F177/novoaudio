"""Núcleo compartilhado das ferramentas do ciclo de qualidade (ferramenta de dev).

Caminhos, manifesto, splits, leitura e escrita de rótulos e o repositório de
rótulos humanos. Os módulos `eval_cycle`, `eval_calibration`, `eval_blind` e
`listen` usam só isto, para não depender uns dos outros.

Splits do manifesto:
- `dev`: o que o analista vê e usa para planejar;
- `holdout`: aceite de cada ciclo; o analista e o implementador não veem;
- `exame`: selado, só nos marcos (portão de fase). Ninguém itera em cima dele.

Rótulos humanos ficam em `evals/human_labels/<par>/<video>.jsonl`,
versionados, com chave (segmento, rev). Continuam válidos em qualquer ciclo
enquanto o segmento não mudar, e substituem o rótulo do juiz onde existirem.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

CYCLES = Path(".cache/cycles")
DUB_CACHE = Path(".cache/dub")
TAXONOMY = Path("evals/taxonomy.json")
HUMAN_LABELS = Path("evals/human_labels")
CALIBRATION = Path("evals/calibration")
JUDGE_AGENT = Path(".claude/agents/dub-judge.md")
SPLITS = ("dev", "holdout")
ALL_SPLITS = ("dev", "holdout", "exame")
PROBLEM_LABELS = ("corrigiria", "escutar")


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def git_state() -> dict[str, Any]:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                    text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return {"git_sha": None, "git_dirty": None}
    return {"git_sha": sha, "git_dirty": dirty}


def judge_model(agent_path: Path = JUDGE_AGENT) -> str | None:
    """Modelo fixado no frontmatter do dub-judge (rótulos de modelos diferentes
    não são comparáveis)."""
    if not agent_path.exists():
        return None
    for line in agent_path.read_text(encoding="utf-8").splitlines()[1:]:
        if line.strip() == "---":
            break
        if line.startswith("model:"):
            return line.split(":", 1)[1].strip()
    return None


# ---------------------------------------------------------------- ciclo


def cycle_dir(cycle_id: str, root: Path = CYCLES) -> Path:
    return root / cycle_id


def load_cycle(cycle_id: str, root: Path = CYCLES) -> tuple[dict[str, Any], dict[str, Any]]:
    """Devolve (cycle.json, manifesto)."""
    meta = read_json(cycle_dir(cycle_id, root) / "cycle.json")
    return meta, read_json(Path(meta["manifest"]))


def pair_name(manifest: dict[str, Any]) -> str:
    return manifest.get("par") or "sem-par"


def shorts(manifest: dict[str, Any], split: str | None = None) -> list[dict[str, Any]]:
    return [s for s in manifest["shorts"] if split is None or s.get("split") == split]


def labels_path(cdir: Path, label: str, split: str, video: str) -> Path:
    return cdir / label / "labels" / split / f"{video}.jsonl"


def run_dir(cdir: Path, label: str, video: str) -> Path:
    return cdir / label / "dub" / video


def read_labels(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Devolve (rótulos de segmento, rótulo do Short ou None)."""
    segments, short = [], None
    for row in read_jsonl(path):
        if "short" in row:
            short = row
        else:
            segments.append(row)
    return segments, short


def write_labels(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")


def read_pending(cdir: Path, label: str, video: str) -> list[str]:
    path = cdir / label / "pending" / f"{video}.txt"
    return path.read_text(encoding="utf-8").split() if path.exists() else []


# ---------------------------------------------------------------- rótulos humanos


def human_path(par: str, video: str, root: Path = HUMAN_LABELS) -> Path:
    return root / par / f"{video}.jsonl"


def load_human(par: str, video: str, root: Path = HUMAN_LABELS
               ) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]]]:
    """Rótulos humanos de um vídeo: {(segmento, rev): rótulo}, {run_rev: veredito}.
    A entrada mais recente vence."""
    segments: dict[tuple[str, str], dict[str, Any]] = {}
    verdicts: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(human_path(par, video, root)):
        if "short" in row:
            verdicts[row.get("run_rev", "")] = row
        else:
            segments[(row["segment_id"], row["rev"])] = row
    return segments, verdicts


def add_human(par: str, row: dict[str, Any], root: Path = HUMAN_LABELS) -> dict[str, Any]:
    row = {**row, "rotulador": "humano", "em": now()}
    append_jsonl(human_path(par, row["video"], root), row)
    return row
