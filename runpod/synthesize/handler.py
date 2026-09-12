"""Worker RunPod Serverless: síntese de fala (MOSS-TTS-v1.5).

job["input"] = {
    "translated_text": str,                 # tradução já pronta (T0.8), sem marcadores ainda
    "target_seconds": float,                # duração alvo TOTAL do segmento
    "output_key": str,
    "pausas_internas": [{"after_word_index": int, "duration": float}, ...],  # opcional
    "original_word_count": int,             # obrigatório se pausas_internas não vier vazio
    "glossary": {"termo": "ipa", ...},      # opcional, T1.8
    "reference_audio_key": str | None,      # opcional, clonagem de voz
    "tolerance": float,                     # opcional, default 0.08 (±8%, on-screen)
}
retorna = {
    "audio_key": str,
    "achieved_seconds": float,
    "tokens_used": int,
    "attempts": int,              # 1 ou 2 — ver packages/pipeline/synthesis.py
    "within_tolerance": bool,
}

A conversão duração->tokens, a injeção de `[pause X.Ys]`, o IPA do
glossário e o cálculo de ajuste da segunda tentativa vivem em
`packages/pipeline/synthesis.py` (lógica pura) — este handler só chama o
modelo e aplica esse resultado.

Processor e modelo carregam uma vez no import, fora de `handler`.
"""

import os
import tempfile

import boto3
import torch
import torchaudio
from transformers import AutoModel, AutoProcessor

import runpod
from packages.pipeline.segmentation import InternalPause
from packages.pipeline.synthesis import (
    adjust_tokens_for_retry,
    apply_ipa_overrides,
    duration_to_tokens,
    inject_pauses,
    is_within_tolerance,
)

torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

_REPO_ID = "OpenMOSS-Team/MOSS-TTS-v1.5"
_DEVICE = "cuda"
# O modelo tende ao pt-PT sem essa tag explícita — ver docs/ptbr.md.
_LANGUAGE = "Portuguese"
_DEFAULT_TOLERANCE = 0.08

_processor = AutoProcessor.from_pretrained(_REPO_ID, trust_remote_code=True)
_processor.audio_tokenizer = _processor.audio_tokenizer.to(_DEVICE)
_model = AutoModel.from_pretrained(
    _REPO_ID, trust_remote_code=True, torch_dtype=torch.bfloat16
).to(_DEVICE)


def _r2_client():
    account_id = os.environ["R2_ACCOUNT_ID"]
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def _download_from_r2(key: str) -> str:
    local_path = os.path.join(tempfile.mkdtemp(), os.path.basename(key))
    _r2_client().download_file(os.environ["R2_BUCKET"], key, local_path)
    return local_path


def _upload_to_r2(local_path: str, key: str) -> str:
    _r2_client().upload_file(local_path, os.environ["R2_BUCKET"], key)
    return key


def _synthesize_once(text: str, tokens: int, reference: list[str] | None) -> tuple[float, str]:
    message = _processor.build_user_message(
        text=text, language=_LANGUAGE, tokens=tokens, reference=reference
    )
    batch = _processor([[message]], mode="generation")
    input_ids = batch["input_ids"].to(_DEVICE)
    attention_mask = batch["attention_mask"].to(_DEVICE)
    outputs = _model.generate(
        input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=4096
    )

    local_output = "/tmp/synth.wav"
    achieved_seconds = 0.0
    for decoded in _processor.decode(outputs):
        audio = decoded.audio_codes_list[0]
        sr = _processor.model_config.sampling_rate
        torchaudio.save(local_output, audio.unsqueeze(0), sr)
        achieved_seconds = audio.shape[-1] / sr
        break
    return achieved_seconds, local_output


def handler(job: dict) -> dict:
    data = job["input"]
    target_seconds = data["target_seconds"]
    output_key = data["output_key"]
    reference_audio_key = data.get("reference_audio_key")
    tolerance = data.get("tolerance", _DEFAULT_TOLERANCE)

    pausas = [
        InternalPause(after_word_index=p["after_word_index"], duration=p["duration"])
        for p in data.get("pausas_internas", [])
    ]
    text = apply_ipa_overrides(data["translated_text"], data.get("glossary", {}))
    text = inject_pauses(text, pausas, data.get("original_word_count", 0))

    reference = [_download_from_r2(reference_audio_key)] if reference_audio_key else None

    tokens = duration_to_tokens(target_seconds)
    achieved_seconds, local_output = _synthesize_once(text, tokens, reference)
    attempts = 1

    if not is_within_tolerance(achieved_seconds, target_seconds, tolerance):
        tokens = adjust_tokens_for_retry(tokens, achieved_seconds, target_seconds)
        achieved_seconds, local_output = _synthesize_once(text, tokens, reference)
        attempts = 2

    _upload_to_r2(local_output, output_key)
    return {
        "audio_key": output_key,
        "achieved_seconds": achieved_seconds,
        "tokens_used": tokens,
        "attempts": attempts,
        "within_tolerance": is_within_tolerance(achieved_seconds, target_seconds, tolerance),
    }


runpod.serverless.start({"handler": handler})
