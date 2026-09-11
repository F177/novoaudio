"""Worker RunPod Serverless: síntese de fala (MOSS-TTS-v1.5).

job["input"] = {
    "text": str,                          # já com [pause X.Ys] e IPA prontos
    "target_seconds": float,
    "output_key": str,
    "reference_audio_key": str | None,    # opcional, clonagem de voz
}
retorna = {"audio_key": str}

`text` já deve trazer os marcadores de pausa e IPA prontos — isso é
responsabilidade de `packages/pipeline/synthesis.py` (T0.9), não daqui.

Processor e modelo carregam uma vez no import, fora de `handler`.
"""

import os
import tempfile

import boto3
import torch
import torchaudio
from transformers import AutoModel, AutoProcessor

import runpod

torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

_REPO_ID = "OpenMOSS-Team/MOSS-TTS-v1.5"
_DEVICE = "cuda"
# O modelo tende ao pt-PT sem essa tag explícita — ver docs/ptbr.md.
_LANGUAGE = "Portuguese"

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


def handler(job: dict) -> dict:
    data = job["input"]
    text = data["text"]
    target_seconds = data["target_seconds"]
    output_key = data["output_key"]
    reference_audio_key = data.get("reference_audio_key")

    tokens = round(target_seconds * 12.5)  # 12.5 Hz, granularidade de 80ms
    reference = [_download_from_r2(reference_audio_key)] if reference_audio_key else None

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
    for decoded in _processor.decode(outputs):
        audio = decoded.audio_codes_list[0]
        torchaudio.save(local_output, audio.unsqueeze(0), _processor.model_config.sampling_rate)
        break

    _upload_to_r2(local_output, output_key)
    return {"audio_key": output_key}


runpod.serverless.start({"handler": handler})
