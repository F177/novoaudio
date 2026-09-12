"""Sintetiza cada frase de um conjunto de calibração com o MOSS-TTS (sem
alvo de duração — deixa o modelo no ritmo natural dele) e mede a duração
real de saída. Parte do T0.6 (calibração de duração do TTS).

Precisa rodar numa máquina com GPU e o MOSS-TTS instalado (ver
`runpod/synthesize/base.Dockerfile` pro setup exato). Não faz parte das
dependências do repo.

Uso:
    python scripts/calibration_measure_durations.py entrada.json saida.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="JSON com as frases (ver calibration_templates.py)")
    parser.add_argument("output_csv", help="Caminho do CSV de resultados")
    args = parser.parse_args()

    import torch
    from transformers import AutoModel, AutoProcessor

    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    repo_id = "OpenMOSS-Team/MOSS-TTS-v1.5"
    device = "cuda"

    with open(args.input, encoding="utf-8") as f:
        phrases = json.load(f)
    print(f"{len(phrases)} frases carregadas")

    print("carregando MOSS-TTS...")
    t0 = time.monotonic()
    processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)
    processor.audio_tokenizer = processor.audio_tokenizer.to(device)
    model = AutoModel.from_pretrained(
        repo_id, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    print(f"  carregado em {time.monotonic() - t0:.1f}s")

    results = []
    errors = 0
    for i, item in enumerate(phrases):
        try:
            t0 = time.monotonic()
            message = processor.build_user_message(text=item["text"], language="Portuguese")
            batch = processor([[message]], mode="generation")
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model.generate(
                input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=4096
            )

            actual_duration = None
            for decoded in processor.decode(outputs):
                audio = decoded.audio_codes_list[0]
                sr = processor.model_config.sampling_rate
                actual_duration = audio.shape[-1] / sr
                break

            elapsed = time.monotonic() - t0
            results.append(
                {
                    **item,
                    "actual_duration_seconds": actual_duration,
                    "gen_time_seconds": round(elapsed, 2),
                }
            )
            print(
                f"[{i + 1}/{len(phrases)}] {item['measured_syllables']} sil -> "
                f"{actual_duration:.2f}s ({elapsed:.1f}s gerando)"
            )
        except Exception as e:  # noqa: BLE001 - coleta de dados, um erro não deve parar o lote
            errors += 1
            print(f"[{i + 1}/{len(phrases)}] ERRO: {e}")
            results.append({**item, "actual_duration_seconds": None, "gen_time_seconds": None})

        if (i + 1) % 20 == 0:
            _write_csv(args.output_csv, results)

    _write_csv(args.output_csv, results)
    print(f"concluído: {len(results)} frases, {errors} erros")


def _write_csv(path: str, results: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
