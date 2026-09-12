
# bookworm (Debian 12), não a tag "slim" genérica: o whisperx pina
# faster-whisper==1.0.0 -> av==11.*, cujo código C usa campos de AVFrame
# (channel_layout/channels) que o FFmpeg do Debian trixie (13, "slim" default
# atual) já removeu. bookworm ainda tem o FFmpeg 5.1.x compatível.
FROM python:3.12-slim-bookworm

WORKDIR /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 pkg-config build-essential \
        libavcodec-dev libavformat-dev libavdevice-dev \
        libavutil-dev libswscale-dev libswresample-dev libavfilter-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY _compat.py .

# Modelo pré-baixado na imagem. A diarização (pyannote) é um modelo com
# licença — precisa de HF_TOKEN só durante o build, passado como secret do
# BuildKit (não fica gravado em nenhuma camada):
#   docker buildx build --secret id=hf_token,env=HF_TOKEN ...
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN=$(cat /run/secrets/hf_token) python -c "\
import os; \
import _compat; \
import whisperx; \
from whisperx.diarize import DiarizationPipeline; \
whisperx.load_model('large-v3', 'cpu', compute_type='int8'); \
whisperx.load_align_model(language_code='en', device='cpu'); \
DiarizationPipeline(use_auth_token=os.environ['HF_TOKEN'], device='cpu')"
