FROM python:3.12-slim

WORKDIR /

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/*

# MOSS-TTS-v1.5 não é um pacote pip simples: o repo traz o código do modelo
# (carregado via trust_remote_code=True). torch/torchaudio/transformers ficam
# atrás do extra "torch-runtime" no pyproject.toml do repo — sem o [torch-runtime]
# o `pip install -e .` sozinho não traz nenhum deles.
RUN git clone --depth 1 https://github.com/OpenMOSS/MOSS-TTS.git /moss-tts
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 \
    -e "/moss-tts[torch-runtime]"
RUN pip install --no-cache-dir runpod boto3

# FlashAttention 2 fica de fora por padrão: exige compilar do zero contra uma
# GPU com compute capability >=8 (Ampere+) e deixa o build frágil/lento. Para
# habilitar: `pip install -e "/moss-tts[torch-runtime,flash-attn]"` acima.

# Modelo pré-baixado na imagem (8B parâmetros — a imagem fica grande, ~16GB+).
RUN python -c "\
from transformers import AutoModel, AutoProcessor; \
repo_id = 'OpenMOSS-Team/MOSS-TTS-v1.5'; \
AutoProcessor.from_pretrained(repo_id, trust_remote_code=True); \
AutoModel.from_pretrained(repo_id, trust_remote_code=True)"

COPY handler.py .
CMD ["python", "-u", "handler.py"]
