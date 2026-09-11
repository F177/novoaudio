"""Worker RunPod Serverless: ASR com timestamps por palavra + diarização.

job["input"] = {"audio_key": str}
retorna       = {"transcript_key": str}

Idioma da fonte fixo em inglês: o produto só faz EN -> pt-BR (ver CLAUDE.md,
"O que NÃO construir"), então não existe parâmetro de idioma aqui.

Os três modelos (ASR, alinhamento, diarização) carregam uma vez no import,
fora de `handler`, pra não recarregar em toda chamada de um worker quente.
"""

import json
import os
import tempfile

import boto3
import whisperx
from whisperx.diarize import DiarizationPipeline

import runpod

_DEVICE = "cuda"
_LANGUAGE = "en"

_asr_model = whisperx.load_model("large-v3", _DEVICE, compute_type="float16", language=_LANGUAGE)
_align_model, _align_metadata = whisperx.load_align_model(language_code=_LANGUAGE, device=_DEVICE)
_diarize_model = DiarizationPipeline(token=os.environ["HF_TOKEN"], device=_DEVICE)


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

    audio = whisperx.load_audio(local_audio)
    result = _asr_model.transcribe(audio, batch_size=16)
    result = whisperx.align(
        result["segments"],
        _align_model,
        _align_metadata,
        audio,
        _DEVICE,
        return_char_alignments=False,
    )
    diarize_segments = _diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    output_key = f"{os.path.splitext(audio_key)[0]}.transcript.json"
    local_output = "/tmp/transcript.json"
    with open(local_output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    _upload_to_r2(local_output, output_key)
    return {"transcript_key": output_key}


runpod.serverless.start({"handler": handler})
