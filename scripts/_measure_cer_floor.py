"""Script pontual (não faz parte do pipeline) pra medir o piso real de CER
em pares (referência, hipótese) julgados manualmente como corretos, usando
dados já em cache de uma rodada real do T0.12. Ver T0.13 em TASKS.md.
"""

import json
from pathlib import Path

from packages.pipeline.quality import character_error_rate

d = Path(
    ".cache/dub/YTDown.com_YouTube_Avengers-Doomsday-Official-Trailer-In-Th_Media_irVNGjRFZGk_004_360p"
)

translations = {
    t["id"]: t["candidates"][0]["text"] if t["candidates"] else None
    for t in json.loads((d / "translations.json").read_text(encoding="utf-8"))
}
eval1 = {e["id"]: e["text"] for e in json.loads((d / "eval.json").read_text(encoding="utf-8"))}
eval2 = {
    e["id"]: e["text"]
    for e in json.loads((d / "eval_retry_result_2.json").read_text(encoding="utf-8"))
}
eval3 = {
    e["id"]: e["text"]
    for e in json.loads((d / "eval_retry_result_3.json").read_text(encoding="utf-8"))
}

# Julgamento manual (lendo o texto): pares onde o conteúdo bate (sem
# repetição, sem corte, sem troca de palavra que mude o sentido) —
# pequenas diferenças de transcrição do Whisper (contração "tá"->"está",
# abreviação lida por extenso) contam como ruído normal, não erro.
KNOWN_GOOD = [
    ("seg0000", eval1),
    ("seg0002", eval1),
    ("seg0002", eval3),
    ("seg0003", eval1),
]

rows = []
for seg_id, evalmap in KNOWN_GOOD:
    ref = translations[seg_id]
    hyp = evalmap[seg_id]
    cer = character_error_rate(ref, hyp)
    rows.append((seg_id, cer, ref, hyp))

rows.sort(key=lambda r: r[1])
for seg_id, cer, _ref, _hyp in rows:
    print(f"{seg_id}: cer={cer:.4f}")

cers = [r[1] for r in rows]
print(f"n={len(cers)} min={min(cers):.4f} max={max(cers):.4f} media={sum(cers)/len(cers):.4f}")
