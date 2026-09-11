FROM python:3.12-slim

WORKDIR /

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Modelo pré-baixado na imagem. A diarização (pyannote) é um modelo com
# licença — precisa de HF_TOKEN só durante o build, passado como secret do
# BuildKit (não fica gravado em nenhuma camada):
#   docker buildx build --secret id=hf_token,env=HF_TOKEN ...
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN=$(cat /run/secrets/hf_token) python -c "\
import os; \
import whisperx; \
from whisperx.diarize import DiarizationPipeline; \
whisperx.load_model('large-v3', 'cpu', compute_type='int8'); \
whisperx.load_align_model(language_code='en', device='cpu'); \
DiarizationPipeline(token=os.environ['HF_TOKEN'], device='cpu')"

COPY handler.py .
CMD ["python", "-u", "handler.py"]
