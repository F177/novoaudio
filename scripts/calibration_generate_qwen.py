"""Gera frases pt-BR naturais via Qwen2.5-7B-Instruct, cobrindo uma faixa de
sílabas controlada — a metade "natural" do conjunto de calibração do T0.6.

Precisa rodar numa máquina com GPU (torch + transformers + accelerate
instalados à parte — não fazem parte das dependências do repo: `pip
install torch transformers accelerate`).

Uso:
    python scripts/calibration_generate_qwen.py saida.json \
        --per-target 9 \
        --targets 10 15 20 28 35 42 50 58 65 70
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from packages.ptbr.syllables import count_syllables

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_TARGETS = [10, 15, 20, 28, 35, 42, 50, 58, 65, 70]

PROMPT_TEMPLATE = """Escreva {per_target} frases distintas em português brasileiro falado, \
naturais, como narração de reportagem de TV (temas variados: notícias, cotidiano, fatos \
gerais). Cada frase deve ter aproximadamente {target} sílabas (pode variar uns 20% pra mais \
ou menos). Frases completamente diferentes entre si em conteúdo e estrutura.

Responda APENAS em JSON, sem markdown: {{"frases": ["...", ...]}}"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="Caminho do JSON de saída")
    parser.add_argument("--per-target", type=int, default=9)
    parser.add_argument("--targets", type=int, nargs="+", default=DEFAULT_TARGETS)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"carregando {MODEL_ID}...")
    t0 = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    print(f"  carregado em {time.monotonic() - t0:.1f}s")

    all_phrases = []
    for target in args.targets:
        prompt = PROMPT_TEMPLATE.format(per_target=args.per_target, target=target)
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.9)
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        match = re.search(r"\{.*\}", response, re.DOTALL)
        if not match:
            print(f"  [alvo {target}] resposta sem JSON, pulando")
            continue
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            print(f"  [alvo {target}] JSON inválido, pulando")
            continue

        for frase in data.get("frases", []):
            all_phrases.append(
                {
                    "text": frase,
                    "source": "qwen",
                    "target_syllables": target,
                    "measured_syllables": count_syllables(frase),
                    "num_pauses": 0,
                    "requested_pause_seconds": 0.0,
                }
            )
        print(f"  [alvo {target}] {len(data.get('frases', []))} frases geradas")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_phrases, f, ensure_ascii=False, indent=2)

    print(f"total: {len(all_phrases)} frases salvas em {args.output}")


if __name__ == "__main__":
    main()
