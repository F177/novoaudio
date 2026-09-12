"""Gera frases pt-BR naturais via Qwen2.5-7B-Instruct, cobrindo uma faixa de
sílabas E uma faixa de registros de fala (não só telejornal — o produto
dubla qualquer vídeo do YouTube: vlog, tutorial, gameplay, review,
comédia...) — a metade "natural" do conjunto de calibração do T0.6.

Precisa rodar numa máquina com GPU (torch + transformers + accelerate
instalados à parte — não fazem parte das dependências do repo: `pip
install torch transformers accelerate`).

Uso:
    python scripts/calibration_generate_qwen.py saida.json \
        --per-combo 2 \
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

REGISTER_DESCRIPTIONS: dict[str, str] = {
    "noticia": "narração de telejornal: tom formal, impessoal, de terceira pessoa",
    "vlog": "vlog pessoal casual: tom informal, primeira pessoa, se dirigindo aos seguidores",
    "tutorial": "tutorial passo a passo: tom instrutivo, se dirigindo ao espectador (\"você\")",
    "gameplay": "comentário de gameplay ao vivo: tom animado, espontâneo, reações no calor",
    "review": "resenha de produto: tom opinativo, avaliando prós e contras",
    "comedia": "comédia/stand-up: tom irônico, exagerado, contando uma história engraçada",
}

PROMPT_TEMPLATE = """Escreva {per_combo} frases em português brasileiro falado, no estilo de \
{register_desc}. Cada frase deve ter aproximadamente {target} sílabas (pode variar uns 20% pra \
mais ou menos). Frases completamente diferentes entre si em conteúdo.

Responda APENAS em JSON, sem markdown: {{"frases": ["...", ...]}}"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="Caminho do JSON de saída")
    parser.add_argument("--per-combo", type=int, default=2)
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
        for register, register_desc in REGISTER_DESCRIPTIONS.items():
            prompt = PROMPT_TEMPLATE.format(
                per_combo=args.per_combo, register_desc=register_desc, target=target
            )
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer([text], return_tensors="pt").to("cuda")
            outputs = model.generate(
                **inputs, max_new_tokens=768, do_sample=True, temperature=0.9
            )
            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )

            match = re.search(r"\{.*\}", response, re.DOTALL)
            if not match:
                print(f"  [{register}/{target}] resposta sem JSON, pulando")
                continue
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                print(f"  [{register}/{target}] JSON inválido, pulando")
                continue

            for frase in data.get("frases", []):
                all_phrases.append(
                    {
                        "text": frase,
                        "source": "qwen",
                        "register": register,
                        "target_syllables": target,
                        "measured_syllables": count_syllables(frase),
                        "num_pauses": 0,
                        "requested_pause_seconds": 0.0,
                    }
                )
            print(f"  [{register}/{target}] {len(data.get('frases', []))} frases geradas")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_phrases, f, ensure_ascii=False, indent=2)

    print(f"total: {len(all_phrases)} frases salvas em {args.output}")


if __name__ == "__main__":
    main()
