"""Worker RunPod Serverless: tradução com restrição de comprimento (Qwen2.5-7B-Instruct).

job["input"] = {
    "source_text": str,
    "target_syllables": int,
    "n_candidates": int,   # opcional, default packages.pipeline.translation.N_CANDIDATES
}
retorna = {"candidates": [{"text", "syllables", "budget_score", "naturalness_score", "score"}, ...]}
         já ordenado do melhor pro pior — ver rank_candidates().

A montagem do prompt, o parsing da resposta e a pontuação local vivem em
packages/pipeline/translation.py (lógica pura, testada sem GPU). Este
handler só carrega o modelo e faz a chamada.

Modelo carrega uma vez no import, fora de `handler`.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import runpod
from packages.pipeline.translation import (
    N_CANDIDATES,
    build_prompt,
    parse_candidates,
    rank_candidates,
)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cuda"

_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)


def handler(job: dict) -> dict:
    data = job["input"]
    source_text = data["source_text"]
    target_syllables = data["target_syllables"]
    n_candidates = data.get("n_candidates", N_CANDIDATES)

    prompt = build_prompt(source_text, target_syllables, n_candidates=n_candidates)
    messages = [{"role": "user", "content": prompt}]
    text = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _tokenizer([text], return_tensors="pt").to(DEVICE)
    outputs = _model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.7)
    response = _tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )

    candidates = parse_candidates(response)
    ranked = rank_candidates(candidates, target_syllables)

    return {
        "candidates": [
            {
                "text": c.text,
                "syllables": c.syllables,
                "budget_score": c.budget_score,
                "naturalness_score": c.naturalness_score,
                "score": c.score,
            }
            for c in ranked
        ]
    }


runpod.serverless.start({"handler": handler})
