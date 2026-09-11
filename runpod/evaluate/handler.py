"""Worker RunPod Serverless: transcrição para CER round-trip (Whisper large-v3).

job["input"] = {"audio_key": str}
retorna       = {"eval_key": str}

Só transcreve o áudio sintetizado (sempre pt-BR). Comparar com o texto alvo
(CER) é lógica de CPU em packages/pipeline/quality.py (T0.11), não roda aqui.

O modelo carrega uma vez no import, fora de `handler`.
"""

import json
import os
import tempfile

import boto3
from faster_whisper import WhisperModel

import runpod

_model = WhisperModel("large-v3", device="cuda", compute_type="float16")


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
    audio_key = job["input"]["audio_key"]
    local_audio = _download_from_r2(audio_key)

    segments, _info = _model.transcribe(local_audio, language="pt", beam_size=5)
    text = " ".join(segment.text.strip() for segment in segments)

    output_key = f"{os.path.splitext(audio_key)[0]}.eval.json"
    local_output = "/tmp/eval.json"
    with open(local_output, "w", encoding="utf-8") as f:
        json.dump({"text": text}, f, ensure_ascii=False)
    _upload_to_r2(local_output, output_key)
    return {"eval_key": output_key}


runpod.serverless.start({"handler": handler})
