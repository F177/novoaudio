"""T0.26 — isolar a causa do piso de 3,2s (`MIN_SYNTHESIS_SECONDS`).

Roda DENTRO do Pod, venv TTS (MOSS-TTS) — mesmo padrão de
`calibration_measure_durations.py`: não faz parte das dependências do
repo, é um harness de coleta de dado real pontual.

Duas curvas, ~10 amostras por ponto (ver TASKS.md T0.26):
  (a) texto fixo (frase de 8 palavras), variando `tokens` em
      {13, 19, 25, 32, 40} — isola o efeito do ORÇAMENTO sobre o mesmo
      conteúdo.
  (b) `tokens` fixo (25, meio da faixa de (a)), variando o texto entre
      prefixos de 1/2/4/8 palavras da MESMA frase — isola o efeito do
      CONTEXTO (frase completa vs. fragmento curto) sobre um orçamento
      igual. Prefixo truncado de propósito, não frase natural reescrita:
      é a mesma situação real de um segmento cortado no meio de uma fala
      (ex. "Here you" do vídeo padrão) que motivou esta investigação.

Gera o áudio de cada amostra e um `manifest.json` (texto de referência,
tokens pedido, duração alcançada, caminho do wav) — a métrica de CER é
calculada depois, localmente, comparando com a transcrição real
(`pod_worker.py evaluate`, venv ASR) via `packages.pipeline.quality`.

Uso (no Pod, venv TTS):
    python scripts/experiment_t026_synthesis_floor.py --out-dir t026_floor
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REFERENCE_SENTENCE = "Ele decide sair de casa agora mesmo rapidinho"
REFERENCE_WORDS = REFERENCE_SENTENCE.split()

CURVE_A_TOKENS = [13, 19, 25, 32, 40]
CURVE_B_WORD_COUNTS = [1, 2, 4, 8]
CURVE_B_TOKENS = 25
N_SAMPLES = 10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    import torch
    import torchaudio
    from transformers import AutoModel, AutoProcessor

    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_id = "OpenMOSS-Team/MOSS-TTS-v1.5"
    device = "cuda"
    print("carregando MOSS-TTS...")
    t0 = time.monotonic()
    processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)
    processor.audio_tokenizer = processor.audio_tokenizer.to(device)
    model = AutoModel.from_pretrained(
        repo_id, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=device
    )
    print(f"  carregado em {time.monotonic() - t0:.1f}s")

    def generate_point(point_id: str, text: str, tokens: int, n: int) -> list[dict]:
        message = processor.build_user_message(
            text=text, language="Portuguese", tokens=tokens, reference=None, instruction=None
        )
        batch = processor([[message]] * n, mode="generation")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        max_new_tokens = max(256, tokens * 4)  # mesmo teto calibrado da T0.24
        t_start = time.monotonic()
        outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens
        )
        elapsed = time.monotonic() - t_start

        sr = processor.model_config.sampling_rate
        entries = []
        for i, decoded in enumerate(processor.decode(outputs)):
            audio = decoded.audio_codes_list[0]
            steps_used = int(outputs[i][1].shape[0])
            achieved_seconds = audio.shape[-1] / sr
            audio_filename = f"{point_id}_s{i:02d}.wav"
            torchaudio.save(str(out_dir / audio_filename), audio.unsqueeze(0), sr)
            entries.append(
                {
                    "id": f"{point_id}_s{i:02d}",
                    "point_id": point_id,
                    "reference_text": text,
                    "word_count": len(text.split()),
                    "tokens": tokens,
                    "sample_index": i,
                    "achieved_seconds": achieved_seconds,
                    "steps_used": steps_used,
                    "audio_file": audio_filename,
                }
            )
        print(
            f"[{point_id}] tokens={tokens} n={n} texto={text!r} -> "
            f"{elapsed:.1f}s pro lote, {[round(e['achieved_seconds'], 2) for e in entries]}"
        )
        return entries

    manifest: list[dict] = []

    # Curva (a): texto fixo, tokens variando.
    for tokens in CURVE_A_TOKENS:
        point_id = f"curveA_tok{tokens:03d}"
        manifest.extend(
            generate_point(point_id, REFERENCE_SENTENCE, tokens, N_SAMPLES)
        )

    # Curva (b): tokens fixo, prefixo da frase variando em tamanho.
    for word_count in CURVE_B_WORD_COUNTS:
        text = " ".join(REFERENCE_WORDS[:word_count])
        point_id = f"curveB_words{word_count:02d}"
        manifest.extend(
            generate_point(point_id, text, CURVE_B_TOKENS, N_SAMPLES)
        )

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"concluído: {len(manifest)} amostras, manifest em {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
