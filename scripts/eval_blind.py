"""A/B cego para o juiz na fase verificar (ferramenta de dev).

O juiz não sabe qual versão é a nova. Para cada Short com segmentos que
mudaram, `prepare` cria `blind/<video>/A` e `B`, sorteados entre base e
candidato, com os segmentos que mudaram no candidato e os da base que se
sobrepõem a eles no tempo. O mapa fica em `blind/mapping.json`, que o guarda
do ciclo esconde do juiz. As duas versões são julgadas na mesma sessão, lado
a lado, o que tira o ruído de sessões diferentes da comparação.

`unblind` devolve os rótulos para `<candidato>`, cria `<base>-par` (a base
com os mesmos trechos rejulgados) e grava as preferências do juiz já
desembaralhadas em `preferencias_juiz.json`.
"""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path
from typing import Any

from scripts.eval_common import (
    CYCLES,
    SPLITS,
    cycle_dir,
    labels_path,
    load_cycle,
    read_json,
    read_jsonl,
    read_labels,
    read_pending,
    run_dir,
    shorts,
    write_json,
    write_labels,
)
from scripts.judge_packet import ReviewSegment, load_run

SMALL_FILES = ("report.json", "segments.json", "synth_jobs.json")
LARGE_FILES = ("vocals.wav", "audio.wav")


def _overlaps(a: ReviewSegment, b: ReviewSegment) -> bool:
    return min(a.end, b.end) - max(a.start, b.start) > 0


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _materialize(src: Path, dst: Path, ids: list[str]) -> None:
    """Cópia mínima de uma execução: metadados, fonte e só os segmentos em julgamento."""
    (dst / "final_segments").mkdir(parents=True, exist_ok=True)
    for name in SMALL_FILES:
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
    for name in LARGE_FILES:
        if (src / name).exists():
            _link_or_copy(src / name, dst / name)
    for seg_id in ids:
        wav = src / "final_segments" / f"{seg_id}.wav"
        if wav.exists():
            shutil.copy2(wav, dst / "final_segments" / wav.name)


def prepare(cycle_id: str, base: str, cand: str, root: Path = CYCLES,
            seed: int | None = None) -> dict[str, int]:
    _, manifest = load_cycle(cycle_id, root)
    cdir = cycle_dir(cycle_id, root)
    blind = cdir / "blind"
    if blind.exists():
        shutil.rmtree(blind)
    rng = random.Random(seed)
    mapping: dict[str, dict[str, str]] = {}
    counts = {"shorts": 0, "segmentos": 0}
    for short in shorts(manifest):
        video, split = short["video"], short.get("split")
        if split not in SPLITS:
            continue
        pending = read_pending(cdir, cand, video)
        if not pending:
            continue
        cand_segs = {s.id: s for s in load_run(run_dir(cdir, cand, video))[1]}
        changed = [cand_segs[i] for i in pending if i in cand_segs]
        base_ids = [s.id for s in load_run(run_dir(cdir, base, video))[1]
                    if any(_overlaps(s, c) for c in changed)]
        sides = [(base, base_ids), (cand, [c.id for c in changed])]
        rng.shuffle(sides)
        mapping[video] = {"A": sides[0][0], "B": sides[1][0], "split": split}
        for letter, (label, ids) in zip("AB", sides, strict=True):
            _materialize(run_dir(cdir, label, video), blind / video / letter, ids)
            (blind / video / f"pending_{letter}.txt").write_text(
                "".join(f"{i}\n" for i in ids), encoding="utf-8")
        counts["shorts"] += 1
        counts["segmentos"] += len(changed)
    write_json(blind / "mapping.json", mapping)
    return counts


def _merge(existing: list[dict[str, Any]], new: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {r["segment_id"]: r for r in existing}
    by_id.update({r["segment_id"]: r for r in new})
    return list(by_id.values())


def unblind(cycle_id: str, base: str, cand: str, root: Path = CYCLES) -> dict[str, Any]:
    _, manifest = load_cycle(cycle_id, root)
    cdir = cycle_dir(cycle_id, root)
    blind = cdir / "blind"
    mapping = read_json(blind / "mapping.json")
    paired = f"{base}-par"
    prefs = {"base": 0, "candidato": 0, "igual": 0}
    for short in shorts(manifest):
        video, split = short["video"], short.get("split")
        if split not in SPLITS:
            continue
        base_rows, base_short = read_labels(labels_path(cdir, base, split, video))
        if video not in mapping:
            write_labels(labels_path(cdir, paired, split, video),
                         base_rows + ([base_short] if base_short else []))
            continue
        letter_of = {mapping[video]["A"]: "A", mapping[video]["B"]: "B"}
        for label, target, existing, existing_short in (
            (base, paired, base_rows, base_short),
            (cand, cand, *read_labels(labels_path(cdir, cand, split, video))),
        ):
            rows, short_row = read_labels(blind / video / f"labels_{letter_of[label]}.jsonl")
            revs = {s.id: s.rev for s in load_run(run_dir(cdir, label, video))[1]}
            fixed = [{**r, "video": video, "rev": revs[r["segment_id"]], "cego": True}
                     for r in rows if r.get("segment_id") in revs]
            merged = _merge(existing, fixed)
            final_short = short_row or existing_short
            if final_short:
                final_short = {**final_short, "video": video}
            write_labels(labels_path(cdir, target, split, video),
                         merged + ([final_short] if final_short else []))
        for pref in read_jsonl(blind / video / "preferencias.jsonl"):
            winner = pref.get("melhor")
            if winner in ("A", "B"):
                prefs["candidato" if mapping[video][winner] == cand else "base"] += 1
            elif winner == "igual":
                prefs["igual"] += 1
    write_json(cdir / "preferencias_juiz.json", prefs)
    return {"rotulo_pareado": paired, "preferencias": prefs}
