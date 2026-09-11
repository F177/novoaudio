"""Worker RunPod Serverless: separação voz/fundo (Demucs htdemucs).

job["input"] = {"source_key": str}
retorna       = {"vocals_key": str, "background_key": str}

O modelo carrega uma vez, no import (fora de `handler`), pra ficar em memória
entre chamadas enquanto o worker estiver quente — só recarrega no cold start.
"""

import os
import tempfile

import boto3
import demucs.api

import runpod


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


_separator = demucs.api.Separator(model="htdemucs", device="cuda")


def handler(job: dict) -> dict:
    source_key = job["input"]["source_key"]
    local_source = _download_from_r2(source_key)

    _original, separated = _separator.separate_audio_file(local_source)
    vocals = separated["vocals"]
    background = separated["drums"] + separated["bass"] + separated["other"]

    stem, _ext = os.path.splitext(source_key)
    vocals_path, background_path = "/tmp/vocals.wav", "/tmp/background.wav"
    demucs.api.save_audio(vocals, vocals_path, samplerate=_separator.samplerate)
    demucs.api.save_audio(background, background_path, samplerate=_separator.samplerate)

    vocals_key = f"{stem}.vocals.wav"
    background_key = f"{stem}.background.wav"
    _upload_to_r2(vocals_path, vocals_key)
    _upload_to_r2(background_path, background_key)
    return {"vocals_key": vocals_key, "background_key": background_key}


runpod.serverless.start({"handler": handler})
