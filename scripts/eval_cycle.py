"""Mecânica determinística do ciclo de qualidade (ferramenta de dev).

Os agentes julgam e propõem; este script copia, conta e compara, para que
nenhum número do ciclo dependa de um LLM somar direito.

    new        --manifest evals/golden/en-pt-BR.json [--slug nome]
    snapshot   <ciclo> <rotulo> [--marco]       copia de .cache/dub/ o que o juiz lê
                                                (dev e holdout; com --marco, só o exame)
    carry      <ciclo> <de> <para>              herda rótulos de segmentos idênticos
                                                (<de> pode ser <outro-ciclo>:<rotulo>)
    summary    <ciclo> <rotulo> [--split dev|holdout] [--marco]
    calibrate  <ciclo> <rotulo> [--repeat <rotulo2>]
    blind      <ciclo> <base> <candidato>       prepara o A/B cego do juiz
    unblind    <ciclo> <base> <candidato>       devolve rótulos e preferências
    compare    <ciclo> <base> <candidato> --alvo classe[,classe]
    usage      --manifest evals/golden/en-pt-BR.json

Layout em .cache/cycles/<ciclo>/:
    cycle.json, compare.json, preferencias_juiz.json
    <rotulo>/dub/<video>/                      snapshot de uma execução
    <rotulo>/labels/<split>/<video>.jsonl      rótulos do dub-judge
    <rotulo>/pending/<video>.txt               segmentos que mudaram e precisam de juiz
    <rotulo>/escuta/<video>.json               fila de escuta sugerida pelo juiz
    blind/                                     A/B cego (mapa escondido do juiz)

Rótulos humanos (`evals/human_labels/`) substituem os do juiz onde existem.
Aceite só sai se: a classe-alvo melhora no dev além do ruído medido do juiz
e em mais Shorts do que piora; não piora no holdout; nada regride (outras
classes, Shorts publicáveis, CER final, fala descartada, custo por Short,
preferência do juiz ou sua); e a classe-alvo está calibrada ou os segmentos
que mudaram têm rótulo humano.

O ruído do juiz é assimétrico de propósito: ele nunca cria um aceite (a
melhora tem que passar dele), mas pode causar rejeição (regressões são
estritas). Rejeição por ruído se resolve rotulando os segmentos com
`scripts.listen`: o rótulo humano substitui o do juiz e tira o ruído.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.eval_blind import prepare as blind_prepare
from scripts.eval_blind import unblind as blind_unblind
from scripts.eval_calibration import (
    calibrate,
    load_calibration,
    merge_human,
    noise_rate,
    render_calibration,
    trusted_classes,
)
from scripts.eval_common import (
    CALIBRATION,
    CYCLES,
    DUB_CACHE,
    HUMAN_LABELS,
    PROBLEM_LABELS,
    SPLITS,
    TAXONOMY,
    append_jsonl,
    cycle_dir,
    git_state,
    judge_model,
    labels_path,
    load_cycle,
    load_human,
    now,
    pair_name,
    read_json,
    read_jsonl,
    read_labels,
    read_pending,
    run_dir,
    shorts,
    write_json,
    write_labels,
)
from scripts.judge_packet import load_run, run_revision

SNAPSHOT_FILES = ("report.json", "segments.json", "synth_jobs.json", "translations.json",
                  "stage_timings.json", "audio.wav", "vocals.wav", "final_audio.wav")
SNAPSHOT_DIRS = ("final_segments",)
ACCEPT_DEFAULTS = {"tolerancia_abs": 1, "tolerancia_rel": 0.1, "cer_final_margem": 0.02,
                   "descarte_margem_rel": 0.1, "custo_margem_rel": 0.1, "custo_margem_abs_s": 2.0}
MAX_HOLDOUT_USES = 8

__all__ = ["carry", "compare", "labels_path", "new_cycle", "read_labels", "snapshot",
           "summarize", "write_labels"]


# ---------------------------------------------------------------- preparação


def new_cycle(manifest_path: Path, slug: str | None, root: Path = CYCLES) -> str:
    cycle_id = f"{datetime.now():%Y%m%d-%H%M}" + (f"-{slug}" if slug else "")
    write_json(cycle_dir(cycle_id, root) / "cycle.json", {
        "id": cycle_id, "manifest": str(manifest_path), "criado_em": now(),
        "juiz_modelo": judge_model(), **git_state(),
    })
    return cycle_id


def snapshot(cycle_id: str, label: str, root: Path = CYCLES, dub_cache: Path = DUB_CACHE,
             splits: tuple[str, ...] = SPLITS) -> list[str]:
    """Copia a execução atual de cada Short dos `splits`; devolve os vídeos sem execução.
    O exame selado fica fora dos ciclos normais: só entra com splits=("exame",), no marco."""
    _, manifest = load_cycle(cycle_id, root)
    base = cycle_dir(cycle_id, root) / label
    missing = []
    for short in [s for s in shorts(manifest) if s.get("split") in splits]:
        src = dub_cache / short["video"]
        if not (src / "report.json").exists():
            missing.append(short["video"])
            continue
        dst = base / "dub" / short["video"]
        dst.mkdir(parents=True, exist_ok=True)
        for name in SNAPSHOT_FILES:
            if (src / name).exists():
                shutil.copy2(src / name, dst / name)
        for name in SNAPSHOT_DIRS:
            if (src / name).is_dir():
                shutil.copytree(src / name, dst / name, dirs_exist_ok=True)
    write_json(base / "meta.json", {"rotulo": label, "faltando": missing, "em": now(),
                                    **git_state()})
    return missing


def carry(cycle_id: str, src: str, dst: str, root: Path = CYCLES) -> dict[str, int]:
    """Herda rótulos de `src` para `dst` onde o segmento é idêntico (mesmo id e
    rev); grava em `dst/pending/` o que mudou e precisa do juiz."""
    _, manifest = load_cycle(cycle_id, root)
    cdir = cycle_dir(cycle_id, root)
    src_cycle, _, src_label = src.rpartition(":")
    src_dir = cycle_dir(src_cycle, root) if src_cycle else cdir
    totals = {"herdados": 0, "pendentes": 0}
    for short in shorts(manifest):
        video, split = short["video"], short["split"]
        rdir = run_dir(cdir, dst, video)
        if not (rdir / "report.json").exists():
            continue
        old_segments, old_short = read_labels(labels_path(src_dir, src_label, split, video))
        old_by_id = {r["segment_id"]: r for r in old_segments}
        carried, pending = [], []
        for seg in load_run(rdir)[1]:
            old = old_by_id.get(seg.id)
            if old is not None and old.get("rev") == seg.rev:
                carried.append({**old, "herdado_de": src})
            else:
                pending.append(seg.id)
        totals["herdados"] += len(carried)
        totals["pendentes"] += len(pending)
        if not pending and old_short is not None:
            carried.append({**old_short, "herdado_de": src})
        write_labels(labels_path(cdir, dst, split, video), carried)
        pending_path = cdir / dst / "pending" / f"{video}.txt"
        pending_path.parent.mkdir(parents=True, exist_ok=True)
        pending_path.write_text("".join(f"{p}\n" for p in pending), encoding="utf-8")
    return totals


# ---------------------------------------------------------------- contagem


def summarize(cycle_id: str, label: str, split: str, root: Path = CYCLES,
              taxonomy_path: Path = TAXONOMY, human_root: Path = HUMAN_LABELS,
              snapshot_label: str | None = None) -> dict[str, Any]:
    """Classes de falha de um rótulo num split, com rótulo humano por cima do juiz."""
    _, manifest = load_cycle(cycle_id, root)
    cdir, par = cycle_dir(cycle_id, root), pair_name(manifest)
    snap = snapshot_label or label
    known = set(read_json(taxonomy_path)["classes"])
    rule = manifest.get("elegibilidade", {"min_shorts": 3, "min_categorias": 2})
    classes: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"corrigiria": 0, "escutar": 0, "shorts": set(), "categorias": set()})
    verdicts: dict[str, int] = defaultdict(int)
    per_short: dict[str, dict[str, int]] = {}
    sources = {"juiz": 0, "humano": 0}
    rejudged, unlabeled, incomplete = 0, [], []
    for short in shorts(manifest, split):
        video = short["video"]
        raw, verdict = read_labels(labels_path(cdir, label, split, video))
        if not raw:
            unlabeled.append(video)
            continue
        human, human_verdicts = load_human(par, video, human_root)
        rows = merge_human(raw, human)
        rdir = run_dir(cdir, snap, video)
        if (rdir / "report.json").exists():
            segs = load_run(rdir)[1]
            if len({r["segment_id"] for r in rows}) < len(segs):
                incomplete.append(video)
            human_verdict = human_verdicts.get(run_revision(segs))
            if human_verdict is not None:
                verdict = human_verdict
        verdicts[(verdict or {}).get("short", "sem_veredito")] += 1
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            sources[row["fonte"]] += 1
            if row["fonte"] == "juiz" and "herdado_de" not in row:
                rejudged += 1
            if row.get("rotulo") not in PROBLEM_LABELS:
                continue
            name = row.get("classe") or "outro"
            entry = classes[name]
            entry[row["rotulo"]] += 1
            entry["shorts"].add(video)
            entry["categorias"].add(short.get("categoria") or "?")
            counts[name] += 1
        per_short[video] = dict(counts)
    table = {
        name: {"corrigiria": e["corrigiria"], "escutar": e["escutar"], "shorts": len(e["shorts"]),
               "categorias": len(e["categorias"]),
               "elegivel": len(e["shorts"]) >= rule["min_shorts"]
               and len(e["categorias"]) >= rule["min_categorias"],
               "na_taxonomia": name in known}
        for name, e in classes.items()
    }
    return {"split": split, "classes": table, "shorts": dict(verdicts), "por_short": per_short,
            "fontes": sources, "reavaliados_pelo_juiz": rejudged, "sem_rotulo": unlabeled,
            "incompletos": incomplete, "regra": rule}


def render_summary(s: dict[str, Any], detail: bool) -> str:
    lines = [f"## {s['split']} — Shorts: " + ", ".join(f"{k} {v}" for k, v in
                                                      sorted(s["shorts"].items())),
             f"rótulos: juiz {s['fontes']['juiz']}, humano {s['fontes']['humano']}"]
    if s["sem_rotulo"]:
        lines.append(f"⚠ sem rótulos: {', '.join(s['sem_rotulo'])}")
    if s["incompletos"]:
        lines.append(f"⚠ rótulos incompletos: {', '.join(s['incompletos'])}")
    ranked = sorted(s["classes"].items(),
                    key=lambda kv: (-kv[1]["shorts"], -kv[1]["corrigiria"] - kv[1]["escutar"]))
    if not detail:
        by_family: dict[str, int] = defaultdict(int)
        for name, c in ranked:
            by_family[name.split(".")[0]] += c["corrigiria"] + c["escutar"]
        lines.append("por família: " + ", ".join(f"{k} {v}" for k, v in sorted(by_family.items())))
        return "\n".join(lines)
    rule = s["regra"]
    lines.append(f"elegível = ≥{rule['min_shorts']} Shorts e ≥{rule['min_categorias']} categorias")
    lines.append("classe | corrigiria | escutar | Shorts | categorias | elegível")
    for name, c in ranked:
        tag = "" if c["na_taxonomia"] else " (fora da taxonomia)"
        lines.append(f"{name}{tag} | {c['corrigiria']} | {c['escutar']} | {c['shorts']} | "
                     f"{c['categorias']} | {'sim' if c['elegivel'] else 'não'}")
    return "\n".join(lines)


# ---------------------------------------------------------------- comparação


def _objective(run_root: Path, videos: list[str]) -> dict[str, Any]:
    cers, discarded, costs = [], 0.0, {}
    for video in videos:
        report_path = run_root / video / "report.json"
        if not report_path.exists():
            continue
        report = read_json(report_path)
        cers += [s["cer_final"] for s in report["segments"] if s.get("cer_final") is not None]
        assembly = report.get("assembly") or {}
        discarded += float(assembly.get("timeline_discarded_samples") or 0)
        discarded += float(assembly.get("mix_discarded_samples") or 0)
        timings = run_root / video / "stage_timings.json"
        if timings.exists():
            costs[video] = sum(float(v) for v in read_json(timings).values())
    return {"cer_final_medio": sum(cers) / len(cers) if cers else None,
            "amostras_descartadas": discarded, "custo_por_short": costs}


def sign_test(deltas: list[int]) -> tuple[int, int, float]:
    """(Shorts que melhoraram, que pioraram, p bilateral do teste do sinal)."""
    better, worse = sum(d < 0 for d in deltas), sum(d > 0 for d in deltas)
    n, k = better + worse, max(better, worse)
    if n == 0:
        return 0, 0, 1.0
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2**n
    return better, worse, min(1.0, 2 * tail)


def _counts(summary: dict[str, Any]) -> dict[str, int]:
    return {k: v["corrigiria"] + v["escutar"] for k, v in summary["classes"].items()}


def _human_coverage(cdir: Path, cand: str, par: str, split_videos: list[tuple[str, str]],
                    human_root: Path) -> tuple[int, int]:
    """(segmentos que mudaram, quantos têm rótulo humano)."""
    changed = covered = 0
    for split, video in split_videos:
        pending = set(read_pending(cdir, cand, video))
        if not pending:
            continue
        human, _ = load_human(par, video, human_root)
        rows, _ = read_labels(labels_path(cdir, cand, split, video))
        for row in rows:
            if row["segment_id"] in pending:
                changed += 1
                covered += (row["segment_id"], row.get("rev", "")) in human
    return changed, covered


def holdout_uses(manifest_path: str, root: Path = CYCLES) -> int:
    uses = 0
    for meta_path in root.glob("*/cycle.json"):
        meta = read_json(meta_path)
        if meta.get("manifest") == manifest_path and (meta_path.parent / "compare.json").exists():
            uses += 1
    return uses


def compare(cycle_id: str, base: str, cand: str, targets: list[str], root: Path = CYCLES,
            taxonomy_path: Path = TAXONOMY, human_root: Path = HUMAN_LABELS,
            calib_root: Path = CALIBRATION) -> dict[str, Any]:
    meta, manifest = load_cycle(cycle_id, root)
    cdir, par = cycle_dir(cycle_id, root), pair_name(manifest)
    tol = {**ACCEPT_DEFAULTS, **manifest.get("aceite", {})}
    paired = f"{base}-par"
    base_eff = paired if (cdir / paired / "labels").exists() else base
    calib = load_calibration(par, meta.get("juiz_modelo"), calib_root)
    trusted = trusted_classes(calib)
    flip, noise_measured = noise_rate(calib)
    r: dict[str, Any] = {"alvo": targets, "base_usada": base_eff, "splits": {}, "regressoes": [],
                         "avisos": [], "ruido": flip, "ruido_medido": noise_measured,
                         "alvo_calibrado": all(t in trusted for t in targets)}
    if calib is None:
        r["avisos"].append("juiz sem calibração: rode /ciclo calibrar")
    elif not calib.get("valida"):
        r["avisos"].append("calibração é de outro modelo de juiz: recalibre")
    if not noise_measured:
        r["avisos"].append(f"ruído do juiz não medido; usando {flip} por padrão")
    for split in SPLITS:
        sb = summarize(cycle_id, base_eff, split, root, taxonomy_path, human_root,
                       snapshot_label=base)
        sc = summarize(cycle_id, cand, split, root, taxonomy_path, human_root)
        if sc["sem_rotulo"] or sc["incompletos"] or sb["sem_rotulo"]:
            r["avisos"].append(f"{split}: rótulos faltando ou incompletos")
        cb, cc = _counts(sb), _counts(sc)
        videos = [s["video"] for s in shorts(manifest, split)]
        ob, oc = _objective(cdir / base / "dub", videos), _objective(cdir / cand / "dub", videos)
        noise_tol = flip * sc["reavaliados_pelo_juiz"]
        target_b = sum(cb.get(t, 0) for t in targets)
        target_c = sum(cc.get(t, 0) for t in targets)
        deltas = [sum(sc["por_short"].get(v, {}).get(t, 0) for t in targets)
                  - sum(sb["por_short"].get(v, {}).get(t, 0) for t in targets) for v in videos]
        better, worse, p_value = sign_test(deltas)
        s = r["splits"][split] = {
            "alvo": (target_b, target_c), "tolerancia_ruido": round(noise_tol, 2),
            "sinal": {"melhoraram": better, "pioraram": worse, "p": round(p_value, 3)},
            "classes": {k: (cb.get(k, 0), cc.get(k, 0)) for k in sorted(set(cb) | set(cc))},
            "publicaria": (sb["shorts"].get("publicaria", 0), sc["shorts"].get("publicaria", 0)),
            "objetivas": {"base": ob, "candidato": oc},
        }
        for name, (b, c) in s["classes"].items():
            if name in targets:
                continue
            if c - b > max(tol["tolerancia_abs"], tol["tolerancia_rel"] * b):
                note = "" if name in trusted else " (juiz não calibrado nesta classe)"
                r["regressoes"].append(f"{split}: {name} {b} → {c}{note}")
        pub_b, pub_c = s["publicaria"]
        if pub_c < pub_b:
            r["regressoes"].append(f"{split}: Shorts publicáveis {pub_b} → {pub_c}")
        cer_b, cer_c = ob["cer_final_medio"], oc["cer_final_medio"]
        if cer_b is not None and cer_c is not None and cer_c > cer_b + tol["cer_final_margem"]:
            r["regressoes"].append(f"{split}: CER final médio {cer_b:.3f} → {cer_c:.3f}")
        dis_b, dis_c = ob["amostras_descartadas"], oc["amostras_descartadas"]
        if dis_c > dis_b * (1 + tol["descarte_margem_rel"]) and dis_c > dis_b:
            r["regressoes"].append(
                f"{split}: fala descartada {dis_b:.0f} → {dis_c:.0f} amostras")
        common = sorted(set(ob["custo_por_short"]) & set(oc["custo_por_short"]))
        if common:
            cost_b = sum(ob["custo_por_short"][v] for v in common) / len(common)
            cost_c = sum(oc["custo_por_short"][v] for v in common) / len(common)
            s["custo_medio_s"] = (round(cost_b, 1), round(cost_c, 1))
            if cost_c > cost_b * (1 + tol["custo_margem_rel"]) and \
                    cost_c - cost_b > tol["custo_margem_abs_s"]:
                r["regressoes"].append(
                    f"{split}: custo por Short {cost_b:.0f} s → {cost_c:.0f} s")
        else:
            r["avisos"].append(f"{split}: custo não medido (stage_timings.json ausente)")
    prefs_path = cdir / "preferencias_juiz.json"
    if prefs_path.exists():
        prefs = read_json(prefs_path)
        r["preferencias_juiz"] = prefs
        if prefs.get("base", 0) > prefs.get("candidato", 0):
            r["regressoes"].append(f"o juiz (cego) preferiu a base: {prefs}")
    human_prefs = [p for p in read_jsonl(human_root / par / "preferencias.jsonl")
                   if p.get("ciclo") == cycle_id]
    if human_prefs:
        tally = {k: sum(p["preferido"] == k for p in human_prefs)
                 for k in ("base", "candidato", "igual")}
        r["preferencias_humanas"] = tally
        if tally["base"] > tally["candidato"]:
            r["regressoes"].append(f"você (cego) preferiu a base: {tally}")
    if not r["alvo_calibrado"]:
        split_videos = [(s["split"], s["video"]) for s in shorts(manifest)
                        if s.get("split") in SPLITS]
        changed, covered = _human_coverage(cdir, cand, par, split_videos, human_root)
        r["cobertura_humana"] = (covered, changed)
    uses = holdout_uses(meta["manifest"], root) + (0 if (cdir / "compare.json").exists() else 1)
    limit = manifest.get("higiene", {}).get("max_usos_holdout", MAX_HOLDOUT_USES)
    if uses > limit:
        r["avisos"].append(f"holdout já usado em {uses} ciclos (limite {limit}): renove-o")
    r["veredito"] = _verdict(r)
    write_json(cdir / "compare.json", {"em": now(), **r})
    return r


def _verdict(r: dict[str, Any]) -> str:
    dev, hold = r["splits"]["dev"], r["splits"]["holdout"]
    (dev_b, dev_c), (hold_b, hold_c) = dev["alvo"], hold["alvo"]
    if any("faltando ou incompletos" in a for a in r["avisos"]):
        return "INCOMPLETO: termine os rótulos antes de decidir"
    if not r["alvo_calibrado"]:
        covered, changed = r["cobertura_humana"]
        if covered < changed:
            return (f"PRECISA DE RÓTULO HUMANO: classe-alvo sem calibração; "
                    f"{changed - covered} de {changed} segmentos que mudaram sem seu rótulo")
    if hold_c > hold_b:
        return "REJEITAR: a classe-alvo piorou no holdout (sinal de overfitting)"
    if r["regressoes"]:
        return "REJEITAR: regressões (veja a lista)"
    if dev_b - dev_c <= dev["tolerancia_ruido"] or dev_c >= dev_b:
        return "INCONCLUSIVO: a melhora no dev não passa do ruído do juiz"
    if dev["sinal"]["melhoraram"] <= dev["sinal"]["pioraram"]:
        return "INCONCLUSIVO: a melhora não aparece em mais Shorts do que a piora"
    if hold_b == 0 and hold_c == 0:
        return ("ACEITAR COM RESSALVA: holdout sem casos da classe-alvo; "
                "generalização não verificada")
    return "ACEITAR, sujeito à escuta humana"


def render_compare(r: dict[str, Any]) -> str:
    lines = [f"# Veredito: {r['veredito']}", f"classe-alvo: {', '.join(r['alvo'])} "
             f"({'calibrada' if r['alvo_calibrado'] else 'não calibrada'}) · base: "
             f"{r['base_usada']} · ruído do juiz: {r['ruido']}"
             f"{'' if r['ruido_medido'] else ' (padrão)'}"]
    if "cobertura_humana" in r:
        lines.append("segmentos que mudaram com rótulo humano: "
                     f"{r['cobertura_humana'][0]}/{r['cobertura_humana'][1]}")
    for key, title in (("preferencias_juiz", "juiz (cego)"), ("preferencias_humanas", "você")):
        if key in r:
            lines.append(f"preferência {title}: {r[key]}")
    for split, s in r["splits"].items():
        (b, c), sg = s["alvo"], s["sinal"]
        pb, pc = s["publicaria"]
        lines += ["", f"## {split}",
                  f"alvo: {b} → {c} (ruído tolerado {s['tolerancia_ruido']}) · Shorts que "
                  f"melhoraram {sg['melhoraram']}, pioraram {sg['pioraram']} (p={sg['p']})",
                  f"Shorts publicáveis: {pb} → {pc}"]
        ob, oc = s["objetivas"]["base"], s["objetivas"]["candidato"]
        if ob["cer_final_medio"] is not None and oc["cer_final_medio"] is not None:
            lines.append(f"CER final médio: {ob['cer_final_medio']:.3f} → "
                         f"{oc['cer_final_medio']:.3f}")
        lines.append(f"fala descartada: {ob['amostras_descartadas']:.0f} → "
                     f"{oc['amostras_descartadas']:.0f} amostras")
        if "custo_medio_s" in s:
            lines.append(f"custo médio por Short: {s['custo_medio_s'][0]} s → "
                         f"{s['custo_medio_s'][1]} s")
        if split == "dev":
            lines += [f"{k}: {x} → {y}" for k, (x, y) in s["classes"].items() if x != y]
    if r["regressoes"]:
        lines += ["", "## Regressões", *[f"- {x}" for x in r["regressoes"]]]
    if r["avisos"]:
        lines += ["", "## Avisos", *[f"- {x}" for x in r["avisos"]]]
    return "\n".join(lines)


# ---------------------------------------------------------------- higiene


def usage_path(manifest_path: Path) -> Path:
    return manifest_path.with_suffix(".usage.jsonl")


def log_exam(cycle_id: str, label: str, root: Path = CYCLES) -> int:
    meta, _ = load_cycle(cycle_id, root)
    path = usage_path(Path(meta["manifest"]))
    append_jsonl(path, {"tipo": "exame", "ciclo": cycle_id, "rotulo": label, "em": now()})
    return sum(1 for row in read_jsonl(path) if row.get("tipo") == "exame")


def exam_result(cycle_id: str, label: str, root: Path = CYCLES,
                human_root: Path = HUMAN_LABELS) -> dict[str, Any]:
    """Taxa de publicação no exame, só com veredito humano (é a métrica do portão)."""
    _, manifest = load_cycle(cycle_id, root)
    cdir, par = cycle_dir(cycle_id, root), pair_name(manifest)
    result = {"publicaria": 0, "nao_publicaria": 0, "sem_veredito_humano": []}
    for short in shorts(manifest, "exame"):
        rdir = run_dir(cdir, label, short["video"])
        if not (rdir / "report.json").exists():
            result["sem_veredito_humano"].append(short["video"])
            continue
        _, verdicts = load_human(par, short["video"], human_root)
        verdict = verdicts.get(run_revision(load_run(rdir)[1]))
        if verdict is None:
            result["sem_veredito_humano"].append(short["video"])
        else:
            result[verdict["short"]] += 1
    return result


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # console cp1252 do Windows quebra nos símbolos
    parser = argparse.ArgumentParser(description="Mecânica do ciclo de qualidade")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("new")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--slug")
    for name in ("snapshot", "calibrate"):
        p = sub.add_parser(name)
        p.add_argument("cycle")
        p.add_argument("label")
        if name == "calibrate":
            p.add_argument("--repeat", help="segundo julgamento dos mesmos Shorts (ruído)")
        else:
            p.add_argument("--marco", action="store_true", help="só o exame selado")
    p = sub.add_parser("carry")
    p.add_argument("cycle")
    p.add_argument("src")
    p.add_argument("dst")
    p = sub.add_parser("summary")
    p.add_argument("cycle")
    p.add_argument("label")
    p.add_argument("--split", choices=SPLITS)
    p.add_argument("--marco", action="store_true", help="resultado do exame selado")
    for name in ("blind", "unblind", "compare"):
        p = sub.add_parser(name)
        p.add_argument("cycle")
        p.add_argument("base")
        p.add_argument("candidate")
        if name == "compare":
            p.add_argument("--alvo", required=True, help="classes-alvo separadas por vírgula")
    p = sub.add_parser("usage")
    p.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    if args.cmd == "new":
        print(new_cycle(args.manifest, args.slug))
    elif args.cmd == "snapshot":
        missing = snapshot(args.cycle, args.label,
                           splits=("exame",) if args.marco else SPLITS)
        if missing:
            print(f"sem execução em .cache/dub/: {', '.join(missing)}")
            sys.exit(1)
        print(f"snapshot '{args.label}' gravado")
    elif args.cmd == "carry":
        print(carry(args.cycle, args.src, args.dst))
    elif args.cmd == "summary" and args.marco:
        uses = log_exam(args.cycle, args.label)
        print(f"exame usado {uses} vez(es) — use só em marcos")
        print(exam_result(args.cycle, args.label))
    elif args.cmd == "summary":
        for split in [args.split] if args.split else list(SPLITS):
            # Holdout sai só agregado: o detalhe por classe orientaria o plano.
            print(render_summary(summarize(args.cycle, args.label, split), split == "dev"),
                  end="\n\n")
    elif args.cmd == "calibrate":
        print(render_calibration(calibrate(args.cycle, args.label, args.repeat)))
    elif args.cmd == "blind":
        print(blind_prepare(args.cycle, args.base, args.candidate))
    elif args.cmd == "unblind":
        print(blind_unblind(args.cycle, args.base, args.candidate))
    elif args.cmd == "compare":
        r = compare(args.cycle, args.base, args.candidate, args.alvo.split(","))
        print(render_compare(r))
        sys.exit(0 if r["veredito"].startswith("ACEITAR") else 1)
    elif args.cmd == "usage":
        uses = holdout_uses(str(args.manifest))
        exams = [row for row in read_jsonl(usage_path(args.manifest)) if row.get("tipo") == "exame"]
        print(f"holdout usado em {uses} ciclo(s); exame usado {len(exams)} vez(es)")


if __name__ == "__main__":
    main()
