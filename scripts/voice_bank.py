"""Gera candidatas para o banco de vozes de amostra pt-BR (T0.31). Roda no Pod
(venv tts):

    python -m scripts.voice_bank --out-dir /root/voice_bank [--lora-adapter-path P]

Para cada descrição em `VOICES` (homens de 20 a 50 anos, timbres variados)
sai, em `out-dir`:
- `<id>_A_vg_pt.wav`: MOSS-VoiceGenerator falando português direto. O model
  card só promete chinês e inglês — medido aqui se o português presta.
- `<id>_design_en.wav`: VoiceGenerator falando inglês (timbre desenhado).
- `<id>_B_*.wav`: MOSS-TTS falando o texto pt-BR com `design_en` como voz de
  referência (timbre do VoiceGenerator, português do MOSS + LoRA).
- `random<n>_C_*.wav`: MOSS-TTS sem referência (sorteia um timbre).
A escolha final é de ouvido (idade não tem medida automática).

Vozes 100% sintéticas: nenhuma gravação de pessoa real entra no banco.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Texto falado pelas candidatas: animado, coloquial, ~10 s.
TEXT_PT = (
    "Cara, você não vai acreditar no que aconteceu hoje! Eu tava saindo de casa e, do nada, "
    "encontrei um amigo que eu não via fazia anos. A gente ficou conversando um tempão."
)
TEXT_EN = (
    "Man, you won't believe what happened today! I was leaving home and, out of nowhere, "
    "I ran into a friend I hadn't seen in years. We ended up talking for ages."
)
TARGET_SECONDS = 10.0
RANDOM_VOICES = 12

_STYLE = "speaking casually and naturally like a Brazilian YouTuber, with a Brazilian accent"
# Só homens por enquanto (decisão do usuário, 2026-10-10), 20 a 50 anos, timbres variados.
VOICES = {
    "h21": f"A 21-year-old young man, bright energetic voice, fast speech, {_STYLE}.",
    "h24": f"A 24-year-old man, light slightly nasal voice, playful and joking, {_STYLE}.",
    "h27": f"A 27-year-old man, relaxed friendly voice, clear and smooth, {_STYLE}.",
    "h29": f"A 29-year-old man, punchy loud voice, hyped gamer streamer, {_STYLE}.",
    "h31": f"A 31-year-old man, deep warm baritone, confident, {_STYLE}.",
    "h34": f"A 34-year-old man, slightly raspy husky voice, laid-back, {_STYLE}.",
    "h37": f"A 37-year-old man, clear medium-pitched voice, enthusiastic presenter, {_STYLE}.",
    "h40": f"A 40-year-old man, soft calm voice, thoughtful podcast host, {_STYLE}.",
    "h43": f"A 43-year-old man, low resonant voice, calm and assertive, {_STYLE}.",
    "h46": f"A 46-year-old man, gravelly mature voice, expressive storyteller, {_STYLE}.",
    "h48": f"A 48-year-old man, bright mature tenor voice, upbeat and talkative, {_STYLE}.",
    "h50": f"A 50-year-old man, rich deep voice, steady and authoritative, {_STYLE}.",
}


def design(out_dir: Path) -> None:
    import torch
    import torchaudio
    from transformers import AutoModel, AutoProcessor

    torch.backends.cuda.enable_cudnn_sdp(False)
    repo_id = "OpenMOSS-Team/MOSS-VoiceGenerator"
    processor = AutoProcessor.from_pretrained(
        repo_id, trust_remote_code=True, normalize_inputs=True
    )
    processor.audio_tokenizer = processor.audio_tokenizer.to("cuda")
    model = AutoModel.from_pretrained(
        repo_id, trust_remote_code=True, attn_implementation="sdpa", torch_dtype=torch.bfloat16
    ).to("cuda").eval()
    rate = processor.model_config.sampling_rate
    for voice_id, description in VOICES.items():
        for name, text in ((f"{voice_id}_A_vg_pt", TEXT_PT), (f"{voice_id}_design_en", TEXT_EN)):
            batch = processor(
                [[processor.build_user_message(text=text, instruction=description)]],
                mode="generation",
            )
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=batch["input_ids"].to("cuda"),
                    attention_mask=batch["attention_mask"].to("cuda"),
                    max_new_tokens=600,  # 48 s a 12,5 Hz: corta loop sem cortar fala
                    # decodificação recomendada no model card do VoiceGenerator
                    audio_temperature=1.5,
                    audio_top_p=0.6,
                    audio_top_k=50,
                    audio_repetition_penalty=1.1,
                )
            audio = processor.decode(outputs)[0].audio_codes_list[0]
            torchaudio.save(str(out_dir / f"{name}.wav"), audio.unsqueeze(0).float().cpu(), rate)
            print(f"[voice_bank] {name}: {audio.shape[-1] / rate:.1f}s")
    del model
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--lora-adapter-path", default=None)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    design(out_dir)

    from scripts.pod_worker import cmd_synthesize

    jobs = [
        {
            "id": f"{voice_id}_B",
            "translated_text": TEXT_PT,
            "target_seconds": TARGET_SECONDS,
            "reference_audio_path": str(out_dir / f"{voice_id}_design_en.wav"),
        }
        for voice_id in VOICES
    ] + [
        {"id": f"random{n:02d}_C", "translated_text": TEXT_PT, "target_seconds": TARGET_SECONDS}
        for n in range(RANDOM_VOICES)
    ]
    jobs_path = out_dir / "synth_jobs.json"
    jobs_path.write_text(json.dumps(jobs, ensure_ascii=False), encoding="utf-8")
    cmd_synthesize(
        argparse.Namespace(
            input=str(jobs_path), out_dir=str(out_dir), lora_adapter_path=args.lora_adapter_path
        )
    )


if __name__ == "__main__":
    main()
