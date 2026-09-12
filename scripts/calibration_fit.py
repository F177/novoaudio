"""Ajusta a regressão de duração do T0.6 a partir do CSV de calibração.

`duracao = a + b*silabas + c*segundos_de_pausa_pedidos`

Depois de ouvir os áudios (ver `scripts/calibration_measure_durations.py`
e `packages/pipeline/calibration.example.json`) e decidir quais linhas são
erro de verdade do modelo (engasgo, corte, artefato) em vez de sinal real,
exclua essas linhas pelo índice (0-based, mesma ordem do CSV) e rode:

    python scripts/calibration_fit.py calibration_results.csv --exclude 12 45 88

Isso só imprime os coeficientes e o R². Pra já escrever o
`packages/pipeline/calibration.json`, passe `--output`:

    python scripts/calibration_fit.py calibration_results.csv \
        --exclude 12 45 88 \
        --output packages/pipeline/calibration.json
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument(
        "--exclude", type=int, nargs="*", default=[], help="Índices de linha (0-based) a excluir"
    )
    parser.add_argument("--output", help="Se dado, escreve o calibration.json com esse resultado")
    parser.add_argument("--calibrated-by", default="")
    args = parser.parse_args()

    with open(args.csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    excluded = set(args.exclude)
    kept = [
        r
        for i, r in enumerate(rows)
        if i not in excluded and r.get("actual_duration_seconds") not in (None, "", "None")
    ]
    print(f"{len(rows)} linhas no CSV, {len(excluded)} excluídas, {len(kept)} usadas no ajuste")

    syll = np.array([float(r["measured_syllables"]) for r in kept])
    pause_s = np.array([float(r["requested_pause_seconds"]) for r in kept])
    dur = np.array([float(r["actual_duration_seconds"]) for r in kept])

    X = np.column_stack([syll, pause_s, np.ones_like(dur)])
    coef, *_ = np.linalg.lstsq(X, dur, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((dur - pred) ** 2))
    ss_tot = float(np.sum((dur - dur.mean()) ** 2))
    r_squared = 1 - ss_res / ss_tot

    b, c, a = coef
    print(f"a (intercepto)      = {a:.4f}")
    print(f"b (por sílaba)       = {b:.4f}")
    print(f"c (por seg. de pausa)= {c:.4f}")
    print(f"R²                   = {r_squared:.4f}")

    short = [(r, p) for r, p in zip(kept, pred, strict=False) if float(r["measured_syllables"]) < 5]
    if short:
        errs = [abs(float(r["actual_duration_seconds"]) - p) for r, p in short]
        print(
            f"\nsegmentos <5 sílabas: {len(short)}, "
            f"erro absoluto médio {sum(errs) / len(errs):.2f}s, "
            f"pior caso {max(errs):.2f}s"
        )

    if args.output:
        result = {
            "formula": "duracao_segundos = a + b * silabas + c * segundos_de_pausa_pedidos",
            "coefficients": {
                "a": round(float(a), 4),
                "b": round(float(b), 4),
                "c": round(float(c), 4),
            },
            "r_squared": round(r_squared, 4),
            "n_samples": len(kept),
            "n_excluded": len(excluded),
            "calibrated_at": date.today().isoformat(),
            "calibrated_by": args.calibrated_by,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nescrito em {args.output}")


if __name__ == "__main__":
    main()
