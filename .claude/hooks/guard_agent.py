#!/usr/bin/env python3
"""Hook PreToolUse dos agentes do ciclo de qualidade.

Transforma regras dos prompts em bloqueio de verdade. Lê o JSON do hook no
stdin e sai com 2 (bloqueia, motivo no stderr) ou 0.

    python3 .claude/hooks/guard_agent.py <papel>   papel: judge | analyst | implementer | reviewer

- todos: nada de GPU/Pod (scripts.dub, pod_runner, pod_worker, ssh, scp,
  runpodctl, --force), git push, nem o servidor de escuta (é do humano).
- judge: só escreve em .cache/; não lê o mapa do A/B cego.
- analyst: só escreve em .cache/; nada do holdout nem do exame (nome de vídeo
  ou caminho). Em .cache/ e em evals/human_labels/ só caminhos de dev, sondas,
  propostas e resumos.
- implementer e reviewer: nada de .cache/ nem dos rótulos humanos (trabalham
  pelo plano, sem ver o conjunto). O implementer não escreve em evals/,
  .claude/, TASKS.md, CLAUDE.md, assets/, runpod/ nem diagnostics.py; o
  reviewer só escreve em tests/.

É uma barreira contra o caminho fácil, não contra má-fé: um script pode
abrir qualquer arquivo por dentro.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

FOR_HUMANS = re.compile(
    r"scripts[./]dub\b|scripts[./]listen\b|pod_runner|pod_worker|\bssh\b|\bscp\b"
    r"|\brunpodctl\b|--force\b|\bgit\s+push\b"
)
WRITE_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}
PATH_CHARS = r"[^\s\"'`;|&()<>,]*"
CACHE_TOKEN = re.compile(rf"\.cache(?:/{PATH_CHARS})?")
HUMAN_TOKEN = re.compile(rf"evals/human_labels(?:/{PATH_CHARS})?")
IMPLEMENTER_PROTECTED = (
    "evals/", ".claude/", "TASKS.md", "CLAUDE.md", "assets/", "runpod/",
    "packages/pipeline/diagnostics.py",
)
ANALYST_FILES = ("propostas.md", "sondas.md", "cycle.json", "meta.json")


def project_root() -> Path:
    return Path(os.environ.get("CLAUDE_PROJECT_DIR") or ".").resolve()


def golden_splits(root: Path) -> dict[str, set[str]]:
    splits: dict[str, set[str]] = {"dev": set(), "holdout": set(), "exame": set()}
    for path in (root / "evals" / "golden").glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for short in data.get("shorts", []):
            splits.setdefault(short.get("split", "dev"), set()).add(short["video"])
    return splits


def relative(path_str: str, root: Path) -> str:
    path = Path(path_str)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(root).as_posix()
        except ValueError:
            return path.as_posix()
    return path.as_posix()


def analyst_cache_ok(token: str, dev: set[str]) -> bool:
    parts = [p for p in token.rstrip("/").split("/") if p]
    if "holdout" in parts or "exame" in parts or "blind" in parts:
        return False
    if "sondas" in parts:
        return True
    for key in ("labels", "dub", "pending", "escuta"):
        if key in parts:
            i = parts.index(key)
            if i + 1 >= len(parts):
                return False
            nxt = parts[i + 1]
            return nxt == "dev" if key == "labels" else nxt.rsplit(".", 1)[0] in dev
    return parts[-1] in ANALYST_FILES or parts[-1].startswith("resumo")


def analyst_human_ok(token: str, dev: set[str]) -> bool:
    name = token.rstrip("/").rsplit("/", 1)[-1]
    return name.endswith(".jsonl") and name.removesuffix(".jsonl") in dev


def check(role: str, tool_name: str, tool_input: dict[str, Any], root: Path) -> str | None:
    """Motivo do bloqueio, ou None se a chamada pode seguir."""
    blob = json.dumps(tool_input, ensure_ascii=False)
    if tool_name == "Bash" and FOR_HUMANS.search(str(tool_input.get("command", ""))):
        return "GPU/Pod, git push e o servidor de escuta ficam com o humano; peça ao orquestrador"
    if tool_name in WRITE_TOOLS:
        target = relative(str(tool_input.get("file_path") or tool_input.get("notebook_path")
                              or ""), root)
        if role in ("judge", "analyst") and not target.startswith(".cache/"):
            return f"{role} só escreve em .cache/ (tentou {target})"
        if role == "implementer" and target.startswith(IMPLEMENTER_PROTECTED):
            return f"{target} é protegido do ciclo; mudanças nele são do humano"
        if role == "reviewer" and not target.startswith("tests/"):
            return f"o revisor só escreve testes em tests/ (tentou {target})"
    if role == "judge" and "mapping.json" in blob:
        return "o mapa do A/B cego é escondido do juiz"
    if role in ("implementer", "reviewer") and (".cache" in blob or "human_labels" in blob):
        return "implementador e revisor não leem o conjunto de avaliação; trabalhe pelo plano"
    if role == "analyst":
        splits = golden_splits(root)
        dev = splits.get("dev", set())
        for name in set().union(*(v for k, v in splits.items() if k != "dev")):
            if re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", blob):
                return "holdout e exame são só do avaliador; analise apenas o dev"
        for token in CACHE_TOKEN.findall(blob):
            if not analyst_cache_ok(token, dev):
                return f"em .cache/ o analista só lê caminhos de dev (bloqueado: {token})"
        for token in HUMAN_TOKEN.findall(blob):
            if not analyst_human_ok(token, dev):
                return f"dos rótulos humanos o analista só lê arquivos de dev (bloqueado: {token})"
    return None


def main() -> None:
    role = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        return
    reason = check(role, data.get("tool_name", ""), data.get("tool_input") or {}, project_root())
    if reason:
        print(f"Bloqueado pelo guarda do ciclo ({role}): {reason}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
