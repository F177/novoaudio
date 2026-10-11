"""Calibração do juiz contra rótulos humanos e medida do ruído do juiz.

O ciclo otimiza o que o juiz diz. Este módulo responde se dá para confiar
nele, classe a classe, em vez de supor:

- concordância juiz × humano por classe, em segmentos com o mesmo `rev`:
  precisão (quando o juiz aponta, você concorda que há problema?) e recall
  (quando você aponta, o juiz tinha apontado?);
- ruído do juiz: o mesmo Short julgado duas vezes. A taxa de discordância
  vira a tolerância do `compare` para os segmentos que foram reavaliados.

Classe com amostra pequena ou concordância baixa é "não confiável": o
`compare` não aceita mudança nela sem rótulo humano nos segmentos que mudaram.
O resultado fica em `evals/calibration/<par>.json` (versionado) e vale só
para o modelo de juiz com que foi medido.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.eval_common import (
    CALIBRATION,
    CYCLES,
    HUMAN_LABELS,
    PROBLEM_LABELS,
    cycle_dir,
    labels_path,
    load_cycle,
    load_human,
    now,
    pair_name,
    read_json,
    read_labels,
    shorts,
    write_json,
)

DEFAULTS = {"min_n": 5, "min_precisao": 0.7, "min_recall": 0.6, "ruido_padrao": 0.15}


def flagged(row: dict[str, Any]) -> bool:
    return row.get("rotulo") in PROBLEM_LABELS


def merge_human(rows: list[dict[str, Any]],
                human: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    """Troca o rótulo do juiz pelo humano onde houver (mesmo segmento e rev)."""
    merged = []
    for row in rows:
        h = human.get((row.get("segment_id", ""), row.get("rev", "")))
        if h is None:
            merged.append({**row, "fonte": row.get("fonte", "juiz")})
        else:
            merged.append({**row, "rotulo": h["rotulo"], "classe": h.get("classe"),
                           "fonte": "humano"})
    return merged


def agreement(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    """Concordância entre dois rotuladores. `pairs` = (avaliado, referência)."""
    tp = fp = fn = tn = class_flips = 0
    per: dict[str, dict[str, int]] = defaultdict(
        lambda: {"juiz": 0, "confirmado": 0, "mesma_classe": 0, "ref": 0, "detectado": 0})
    for judged, ref in pairs:
        j, r = flagged(judged), flagged(ref)
        tp, fp, fn, tn = tp + (j and r), fp + (j and not r), fn + (r and not j), tn + (not (j or r))
        if j and r and judged.get("classe") != ref.get("classe"):
            class_flips += 1
        if j:
            c = per[judged.get("classe") or "outro"]
            c["juiz"] += 1
            c["confirmado"] += r
            c["mesma_classe"] += r and judged.get("classe") == ref.get("classe")
        if r:
            c = per[ref.get("classe") or "outro"]
            c["ref"] += 1
            c["detectado"] += j
    n = tp + fp + fn + tn
    if n == 0:
        return {"n": 0, "classes": {}}
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n**2
    return {
        "n": n, "vp": tp, "fp": fp, "fn": fn, "vn": tn,
        "concordancia": round(po, 3),
        "kappa": round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0,
        "discordancia": round((fp + fn + class_flips) / n, 3),
        "classes": {k: dict(v) for k, v in per.items()},
    }


def judge_human_pairs(cycle_id: str, label: str, root: Path = CYCLES,
                      human_root: Path = HUMAN_LABELS) -> list[tuple[dict, dict]]:
    _, manifest = load_cycle(cycle_id, root)
    cdir, par = cycle_dir(cycle_id, root), pair_name(manifest)
    pairs = []
    for short in shorts(manifest):
        rows, _ = read_labels(labels_path(cdir, label, short["split"], short["video"]))
        human, _ = load_human(par, short["video"], human_root)
        pairs += [(r, human[(r["segment_id"], r.get("rev", ""))]) for r in rows
                  if (r["segment_id"], r.get("rev", "")) in human]
    return pairs


def judge_judge_pairs(cycle_id: str, label_a: str, label_b: str,
                      root: Path = CYCLES) -> list[tuple[dict, dict]]:
    _, manifest = load_cycle(cycle_id, root)
    cdir = cycle_dir(cycle_id, root)
    pairs = []
    for short in shorts(manifest):
        a, _ = read_labels(labels_path(cdir, label_a, short["split"], short["video"]))
        b, _ = read_labels(labels_path(cdir, label_b, short["split"], short["video"]))
        b_by_key = {(r["segment_id"], r.get("rev", "")): r for r in b}
        pairs += [(r, b_by_key[(r["segment_id"], r.get("rev", ""))]) for r in a
                  if (r["segment_id"], r.get("rev", "")) in b_by_key]
    return pairs


def class_table(agr: dict[str, Any], rule: dict[str, Any]) -> dict[str, dict[str, Any]]:
    table = {}
    for name, c in agr.get("classes", {}).items():
        precision = c["confirmado"] / c["juiz"] if c["juiz"] else None
        recall = c["detectado"] / c["ref"] if c["ref"] else None
        table[name] = {
            "juiz": c["juiz"], "humano": c["ref"],
            "precisao": round(precision, 2) if precision is not None else None,
            "recall": round(recall, 2) if recall is not None else None,
            "mesma_classe": round(c["mesma_classe"] / c["confirmado"], 2)
            if c["confirmado"] else None,
            "confiavel": bool(c["juiz"] >= rule["min_n"] and c["ref"] >= rule["min_n"]
                              and precision is not None and precision >= rule["min_precisao"]
                              and recall is not None and recall >= rule["min_recall"]),
        }
    return table


def calibrate(cycle_id: str, label: str, repeat: str | None = None, root: Path = CYCLES,
              human_root: Path = HUMAN_LABELS, calib_root: Path = CALIBRATION) -> dict[str, Any]:
    meta, manifest = load_cycle(cycle_id, root)
    par = pair_name(manifest)
    rule = {**DEFAULTS, **manifest.get("calibracao", {})}
    agr = agreement(judge_human_pairs(cycle_id, label, root, human_root))
    path = calib_root / f"{par}.json"
    previous = read_json(path) if path.exists() else {}
    same_model = previous.get("juiz_modelo") == meta.get("juiz_modelo")
    noise = previous.get("ruido") if same_model else None
    if repeat:
        rep = agreement(judge_judge_pairs(cycle_id, label, repeat, root))
        noise = {"discordancia": rep.get("discordancia"), "n": rep["n"], "ciclo": cycle_id,
                 "rotulos": [label, repeat]}
    result = {
        "par": par, "juiz_modelo": meta.get("juiz_modelo"), "atualizado_em": now(),
        "ciclo": cycle_id, "regra": rule,
        "concordancia": {k: v for k, v in agr.items() if k != "classes"},
        "classes": class_table(agr, rule), "ruido": noise,
    }
    write_json(path, result)
    return result


def load_calibration(par: str, model: str | None,
                     calib_root: Path = CALIBRATION) -> dict[str, Any] | None:
    """Calibração do par, marcada `valida` só se foi medida com o mesmo modelo de juiz."""
    path = calib_root / f"{par}.json"
    if not path.exists():
        return None
    data = read_json(path)
    data["valida"] = data.get("juiz_modelo") == model and model is not None
    return data


def trusted_classes(calib: dict[str, Any] | None) -> set[str]:
    if not calib or not calib.get("valida"):
        return set()
    return {k for k, v in calib.get("classes", {}).items() if v.get("confiavel")}


def noise_rate(calib: dict[str, Any] | None) -> tuple[float, bool]:
    """(taxa de discordância do juiz, se foi medida)."""
    if calib and calib.get("valida") and calib.get("ruido") and \
            calib["ruido"].get("discordancia") is not None:
        return float(calib["ruido"]["discordancia"]), True
    return DEFAULTS["ruido_padrao"], False


def render_calibration(c: dict[str, Any]) -> str:
    agr = c["concordancia"]
    lines = [f"# Calibração do juiz — {c['par']} — modelo {c['juiz_modelo']}"]
    if agr.get("n"):
        lines.append(f"segmentos com rótulo humano: {agr['n']} · concordância "
                     f"{agr['concordancia']} · kappa {agr['kappa']}")
    else:
        lines.append("nenhum segmento com rótulo humano e rev igual: rotule com scripts.listen")
    noise = c.get("ruido")
    lines.append(f"ruído do juiz: {noise['discordancia']} ({noise['n']} segmentos)" if noise
                 else "ruído do juiz: não medido (julgue o mesmo Short duas vezes)")
    lines.append("classe | juiz | humano | precisão | recall | confiável")
    for name, v in sorted(c["classes"].items(), key=lambda kv: -kv[1]["humano"]):
        lines.append(f"{name} | {v['juiz']} | {v['humano']} | {v['precisao']} | {v['recall']} | "
                     f"{'sim' if v['confiavel'] else 'não'}")
    return "\n".join(lines)
