# Build context é a RAIZ DO REPO (não runpod/translate/), porque este
# worker importa código puro de packages/pipeline e packages/ptbr:
#   docker buildx build -f runpod/translate/base.Dockerfile .
FROM python:3.12-slim

WORKDIR /

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY runpod/translate/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY packages/__init__.py /packages/__init__.py
COPY packages/pipeline/__init__.py /packages/pipeline/__init__.py
COPY packages/pipeline/translation.py /packages/pipeline/translation.py
COPY packages/ptbr/__init__.py /packages/ptbr/__init__.py
COPY packages/ptbr/syllables.py /packages/ptbr/syllables.py

# Modelo pré-baixado na imagem.
RUN python -c "\
from transformers import AutoModelForCausalLM, AutoTokenizer; \
repo_id = 'Qwen/Qwen2.5-7B-Instruct'; \
AutoTokenizer.from_pretrained(repo_id); \
AutoModelForCausalLM.from_pretrained(repo_id)"
