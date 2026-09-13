"""Ajusta a regressão de duração do T0.6/T0.15 a partir do CSV de calibração.

Três variantes do termo de pausa (`--variant`), comparadas em T0.15 porque
o achado do T0.9 (pausa conta DENTRO do orçamento de tokens, não soma por
cima) invalidou como o termo original era usado: `c=5.6865` foi ajustado
contra pausa medida em geração LIVRE (sem orçamento fechado, mais pausa só
alongava a geração) mas é consumido em `budget.py` como se uma pausa de 1s
custasse 5,68s do orçamento TOTAL do segmento — suficiente pra zerar
`syllables_that_fit` em segmentos comuns (`syllables_that_fit(4.0,
pause_seconds=1.0)` já dá negativo antes de dividir por `b`).

- `pause_seconds` (original): `duracao = a + b*silabas + c*segundos_de_pausa`
- `no_pause`: `duracao = a + b*silabas`, ajustado SÓ nas linhas com
  `num_pauses == 0`. Não é só tirar a coluna de pausa do ajuste — linhas
  com pausa, em geração livre, têm duração inflada pelo tempo real da
  pausa (que "sílabas" sozinho não explica), então incluí-las distorce
  `a,b` (testado: incluir todas as linhas dá R² pior que excluir as com
  pausa). O chamador reserva o tempo de pausa à parte, 1:1, fora da
  fórmula — é isso que T0.9 confirmou que o MOSS-TTS faz no regime de
  orçamento fechado.
- `pause_count`: `duracao = a + b*silabas + c*numero_de_pausas` — captura
  um custo FIXO por marcador de pausa (ex.: overhead de tokenização do
  `[pause X.Ys]`), não escalado pela duração da pausa.

Depois de ouvir os áudios (ver `scripts/calibration_measure_durations.py`
e `packages/pipeline/calibration.example.json`) e decidir quais linhas são
erro de verdade do modelo (engasgo, corte, artefato) em vez de sinal real,
exclua essas linhas pelo índice (0-based, mesma ordem do CSV) e rode:

    python scripts/calibration_fit.py calibration_results.csv --exclude 12 45 88

Isso só imprime os coeficientes e o R². Pra já escrever o
`packages/pipeline/calibration.json`, passe `--output`:

    python scripts/calibration_fit.py calibration_results.csv \
        --exclude 12 45 88 --variant no_pause \
        --output packages/pipeline/calibration.json
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date

import numpy as np

VARIANTS = ("pause_seconds", "no_pause", "pause_count")


def _load_kept_rows(csv_path: str, excluded: set[int]) -> list[dict]:
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    kept = [
        r
        for i, r in enumerate(rows)
        if i not in excluded and r.get("actual_duration_seconds") not in (None, "", "None")
    ]
    print(f"{len(rows)} linhas no CSV, {len(excluded)} excluídas, {len(kept)} usadas no ajuste")
    return kept


def fit_variant(kept: list[dict], variant: str) -> dict:
    """Ajusta uma variante e devolve coeficientes + R² + previsão por linha.

    `no_pause` filtra pra `num_pauses == 0` antes de ajustar — ver docstring
    do módulo pra por quê (evita contaminar `a,b` com linhas cuja duração
    medida inclui tempo de pausa que a fórmula não modela).
    """
    if variant == "no_pause":
        kept = [r for r in kept if float(r["num_pauses"]) == 0]

    syll = np.array([float(r["measured_syllables"]) for r in kept])
    dur = np.array([float(r["actual_duration_seconds"]) for r in kept])

    if variant == "no_pause":
        X = np.column_stack([syll, np.ones_like(dur)])
    elif variant == "pause_count":
        pause_n = np.array([float(r["num_pauses"]) for r in kept])
        X = np.column_stack([syll, pause_n, np.ones_like(dur)])
    elif variant == "pause_seconds":
        pause_s = np.array([float(r["requested_pause_seconds"]) for r in kept])
        X = np.column_stack([syll, pause_s, np.ones_like(dur)])
    else:
        raise ValueError(f"variant desconhecida: {variant}")

    coef, *_ = np.linalg.lstsq(X, dur, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((dur - pred) ** 2))
    ss_tot = float(np.sum((dur - dur.mean()) ** 2))
    r_squared = 1 - ss_res / ss_tot

    if variant == "no_pause":
        b, a = coef
        c = None
    else:
        b, c, a = coef

    return {
        "a": float(a),
        "b": float(b),
        "c": c,
        "r_squared": r_squared,
        "pred": pred,
        "kept": kept,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument(
        "--exclude", type=int, nargs="*", default=[], help="Índices de linha (0-based) a excluir"
    )
    parser.add_argument("--variant", choices=VARIANTS, default="pause_seconds")
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Ajusta as três variantes e só imprime a comparação de R² (ignora --variant)",
    )
    parser.add_argument("--output", help="Se dado, escreve o calibration.json com esse resultado")
    parser.add_argument("--calibrated-by", default="")
    args = parser.parse_args()

    kept = _load_kept_rows(args.csv_path, set(args.exclude))

    if args.compare_all:
        for variant in VARIANTS:
            result = fit_variant(kept, variant)
            c_str = f"{result['c']:.4f}" if result["c"] is not None else "—"
            print(
                f"{variant:15s} a={result['a']:.4f} b={result['b']:.4f} "
                f"c={c_str} R²={result['r_squared']:.4f}"
            )
        return

    result = fit_variant(kept, args.variant)
    a, b, c, r_squared, pred, used = (
        result["a"],
        result["b"],
        result["c"],
        result["r_squared"],
        result["pred"],
        result["kept"],
    )
    print(f"a (intercepto)      = {a:.4f}")
    print(f"b (por sílaba)       = {b:.4f}")
    if c is not None:
        label = "por nº de pausas" if args.variant == "pause_count" else "por seg. de pausa"
        print(f"c ({label}) = {c:.4f}")
    print(f"R²                   = {r_squared:.4f}")
    print(f"n usadas no ajuste   = {len(used)}")

    short = [(r, p) for r, p in zip(used, pred, strict=True) if float(r["measured_syllables"]) < 5]
    if short:
        errs = [abs(float(r["actual_duration_seconds"]) - p) for r, p in short]
        print(
            f"\nsegmentos <5 sílabas: {len(short)}, "
            f"erro absoluto médio {sum(errs) / len(errs):.2f}s, "
            f"pior caso {max(errs):.2f}s"
        )

    if args.output:
        formulas = {
            "pause_seconds": "duracao_segundos = a + b * silabas + c * segundos_de_pausa_pedidos",
            "no_pause": "duracao_segundos = a + b * silabas",
            "pause_count": "duracao_segundos = a + b * silabas + c * numero_de_pausas",
        }
        coefficients = {"a": round(float(a), 4), "b": round(float(b), 4)}
        if c is not None:
            coefficients["c"] = round(float(c), 4)
        result_json = {
            "formula": formulas[args.variant],
            "variant": args.variant,
            "coefficients": coefficients,
            "r_squared": round(r_squared, 4),
            "n_samples": len(used),
            "n_excluded": len(args.exclude),
            "calibrated_at": date.today().isoformat(),
            "calibrated_by": args.calibrated_by,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result_json, f, ensure_ascii=False, indent=2)
        print(f"\nescrito em {args.output}")


if __name__ == "__main__":
    main()
