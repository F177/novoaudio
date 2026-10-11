"""Construtores de ciclos sintéticos para os testes das ferramentas do ciclo."""

import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.eval_common import labels_path, write_labels
from scripts.eval_cycle import new_cycle
from scripts.judge_packet import load_run

SR = 8000


def make_cycle(tmp_path: Path, shorts: list[tuple[str, str, str]],
               **extra: object) -> tuple[Path, str]:
    manifest = {"par": "en-xx", "idioma_alvo": "xx",
                "shorts": [{"video": v, "split": sp, "categoria": c} for v, sp, c in shorts],
                "elegibilidade": {"min_shorts": 2, "min_categorias": 2}, **extra}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    root = tmp_path / "cycles"
    return root, new_cycle(path, "t", root)


def _tone(text: str, seconds: float) -> np.ndarray:
    hz = 100 + int(hashlib.sha1(text.encode()).hexdigest()[:4], 16) % 200
    t = np.arange(int(seconds * SR)) / SR
    return (0.2 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def make_run(rdir: Path, texts: list[str], timings: dict[str, float] | None = None) -> None:
    """Execução sintética: 1 s por segmento, áudio derivado do texto (texto novo = rev novo)."""
    (rdir / "final_segments").mkdir(parents=True, exist_ok=True)
    segs = [{"id": f"seg{i:04d}", "t_inicio": float(i), "t_fim": i + 1.0,
             "texto_original": f"source {i}", "traducao": t, "cer_final": 0.05}
            for i, t in enumerate(texts)]
    (rdir / "report.json").write_text(json.dumps({"segments": segs}), encoding="utf-8")
    full = np.concatenate([_tone(t, 1.0) for t in texts])
    sf.write(rdir / "final_audio.wav", full, SR)
    sf.write(rdir / "audio.wav", _tone("source", len(texts) * 1.0), SR)
    for seg, text in zip(segs, texts, strict=True):
        sf.write(rdir / "final_segments" / f"{seg['id']}.wav", _tone(text, 1.0), SR)
    if timings is not None:
        (rdir / "stage_timings.json").write_text(json.dumps(timings), encoding="utf-8")


def judge_labels(cdir: Path, label: str, split: str, video: str, problems: dict[str, str],
                 rejudged: set[str] | None = None, short: str = "publicaria",
                 run_label: str | None = None) -> None:
    """Rótulos como o juiz escreveria; segmentos fora de `rejudged` contam como herdados."""
    segs = load_run(cdir / (run_label or label) / "dub" / video)[1]
    rows = []
    for s in segs:
        row = {"video": video, "segment_id": s.id, "rev": s.rev,
               "rotulo": "corrigiria" if s.id in problems else "ok",
               "classe": problems.get(s.id), "rotulador": "dub-judge"}
        if rejudged is not None and s.id not in rejudged:
            row["herdado_de"] = "base"
        rows.append(row)
    write_labels(labels_path(cdir, label, split, video),
                 [*rows, {"video": video, "short": short, "rotulador": "dub-judge"}])
