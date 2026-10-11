"""Guarda contra overfitting no diff de uma mudança (ferramenta de dev).

O produto recebe vídeos arbitrários. Uma correção que só funciona porque
conhece o conjunto de avaliação é inútil e esconde o problema. Este script
procura, nas linhas adicionadas desde `--base`:

- bloqueia: trecho do conjunto (n-grama de NGRAM palavras do inglês ou da
  tradução), nome de vídeo do conjunto, id de segmento (`seg` + 4 dígitos);
- avisa: constante numérica de módulo alterada em `packages/` (limiar). Não
  é proibido, mas precisa de justificativa medida no conjunto inteiro.

Os dados de avaliação em si (manifestos, rótulos humanos, calibração e o
ledger) ficam fora da varredura: são do humano e protegidos pelo guarda.

Em `tests/` os achados viram aviso: o projeto usa dado sintético em teste,
com exceção documentada de fixture real.

Com `--dados`, checa um dataset de treino (LoRA, fine-tune) em vez do diff:
treinar com texto do conjunto de avaliação é o overfitting definitivo, porque
o ciclo diria ACEITAR sem o produto melhorar.

Rode depois de commitar (arquivos novos não rastreados também são lidos):
    python -m scripts.check_generalization [--base master]
    python -m scripts.check_generalization --dados dataset/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

NGRAM = 5
DUB_CACHE = Path(".cache/dub")
DUBLADOS = Path("videos/dublados")
GOLDEN_DIR = Path("evals/golden")
SEGMENT_ID = re.compile(r"\bseg\d{4}\b")
EVAL_DATA = ("evals/golden/", "evals/human_labels/", "evals/calibration/", "evals/ledger.md")
CONSTANT = re.compile(r"^(_?[A-Z][A-Z0-9_]*)\s*(?::[^=]+)?=\s*([-+]?\d[\d_.eE+-]*)\s*(?:#.*)?$")


@dataclass(frozen=True)
class Finding:
    severity: str  # "bloqueia" | "aviso"
    path: str
    detail: str


def words(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", text.lower())


def ngrams(tokens: list[str], n: int = NGRAM) -> set[tuple[str, ...]]:
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def parse_diff(diff: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Linhas adicionadas e removidas de um `git diff --unified=0`, com o arquivo."""
    added, removed, path = [], [], ""
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else line[4:]
        elif line.startswith("--- "):
            continue
        elif line.startswith("+"):
            added.append((path, line[1:]))
        elif line.startswith("-"):
            removed.append((path, line[1:]))
    return added, removed


def load_corpus(manifest_paths: list[Path], dub_cache: Path = DUB_CACHE,
                dublados: Path = DUBLADOS) -> tuple[set[tuple[str, ...]], set[str]]:
    """N-gramas de todo texto do conjunto (fonte e tradução) e nomes dos vídeos."""
    grams: set[tuple[str, ...]] = set()
    names: set[str] = set()
    for manifest_path in manifest_paths:
        for short in json.loads(manifest_path.read_text(encoding="utf-8"))["shorts"]:
            video = short["video"]
            names.add(video)
            texts: list[str] = []
            report = dub_cache / video / "report.json"
            if report.exists():
                for seg in json.loads(report.read_text(encoding="utf-8"))["segments"]:
                    texts += [seg.get("texto_original") or "", seg.get("traducao") or ""]
            for path in dublados.glob(f"{video}*_traducao.json"):
                for seg in json.loads(path.read_text(encoding="utf-8")):
                    texts += [seg.get("texto_original") or "", seg.get("traducao_usada") or ""]
            for text in texts:
                grams |= ngrams(words(text))
    return grams, names


def find_leaks(added: list[tuple[str, str]], grams: set[tuple[str, ...]],
               names: set[str]) -> list[Finding]:
    findings = []
    name_patterns = {n: re.compile(rf"(?<![\w-]){re.escape(n)}(?![\w-])") for n in names}
    for path, line in added:
        if path.startswith(EVAL_DATA):
            continue
        severity = "aviso" if path.startswith("tests/") else "bloqueia"
        hit = ngrams(words(line)) & grams
        if hit:
            phrase = " ".join(sorted(hit)[0])
            findings.append(Finding(severity, path, f"texto do conjunto de avaliação: '{phrase}…'"))
        for name, pattern in name_patterns.items():
            if pattern.search(line):
                findings.append(Finding(severity, path, f"nome de vídeo do conjunto: {name}"))
        if SEGMENT_ID.search(line) and not path.startswith("tests/"):
            findings.append(Finding("bloqueia", path, f"id de segmento no código: {line.strip()}"))
    return findings


def threshold_changes(added: list[tuple[str, str]],
                      removed: list[tuple[str, str]]) -> list[Finding]:
    def constants(rows: list[tuple[str, str]]) -> dict[tuple[str, str], str]:
        found = {}
        for path, line in rows:
            match = CONSTANT.match(line.strip())
            if path.startswith("packages/") and match:
                found[(path, match.group(1))] = match.group(2)
        return found

    old, new = constants(removed), constants(added)
    return [
        Finding("aviso", path, f"limiar {name}: {old.get((path, name), 'novo')} → {value} "
                               "(justifique com medida no conjunto inteiro)")
        for (path, name), value in new.items()
        if old.get((path, name)) != value
    ]


def scan_dataset(paths: list[Path], grams: set[tuple[str, ...]]) -> list[Finding]:
    """Linhas de um dataset (jsonl, txt, csv) que contêm texto do conjunto."""
    findings = []
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for number, line in enumerate(fh, start=1):
                hit = ngrams(words(line)) & grams
                if hit:
                    phrase = " ".join(sorted(hit)[0])
                    findings.append(Finding("bloqueia", f"{path}:{number}",
                                            f"texto do conjunto de avaliação: '{phrase}…'"))
    return findings


def default_base() -> str:
    """Branch padrão do repositório (origin/HEAD), senão main ou master."""
    head = subprocess.run(["git", "symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD"],
                          capture_output=True, text=True)
    if head.returncode == 0 and head.stdout.strip():
        return head.stdout.strip().removeprefix("origin/")
    for name in ("main", "master"):
        if subprocess.run(["git", "rev-parse", "--verify", "--quiet", name],
                          capture_output=True).returncode == 0:
            return name
    return "HEAD"


def collect_diff(base: str) -> str:
    proc = subprocess.run(["git", "diff", "--unified=0", base], capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"git diff contra '{base}' falhou: {proc.stderr.strip()} "
                 "(passe o branch base certo com --base)")
    diff = proc.stdout
    untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"],
                               capture_output=True, text=True, check=True).stdout.split()
    for path in untracked:
        try:
            body = Path(path).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        diff += f"\n+++ b/{path}\n" + "".join(f"+{line}\n" for line in body.splitlines())
    return diff


def main() -> None:
    parser = argparse.ArgumentParser(description="Guarda contra overfitting no diff")
    parser.add_argument("--base", help="branch base (padrão: o branch padrão do repo)")
    parser.add_argument("--manifest", type=Path, action="append",
                        help="manifesto do conjunto (padrão: todos em evals/golden/)")
    parser.add_argument("--dados", type=Path, action="append",
                        help="dataset de treino a checar em vez do diff")
    args = parser.parse_args()

    manifests = args.manifest or sorted(p for p in GOLDEN_DIR.glob("*.json")
                                        if not p.name.endswith(".usage.json"))
    grams, names = load_corpus(manifests)
    if args.dados:
        findings = scan_dataset(args.dados, grams)
    else:
        added, removed = parse_diff(collect_diff(args.base or default_base()))
        findings = find_leaks(added, grams, names) + threshold_changes(added, removed)
    for f in findings:
        print(f"[{f.severity}] {f.path}: {f.detail}")
    if not grams:
        print("[aviso] nenhum texto do conjunto encontrado em .cache/dub/ ou videos/dublados/: "
              "a checagem de texto não teve contra o que comparar")
    blocked = [f for f in findings if f.severity == "bloqueia"]
    print(f"{len(blocked)} bloqueio(s), {len(findings) - len(blocked)} aviso(s)")
    sys.exit(1 if blocked else 0)


if __name__ == "__main__":
    main()
