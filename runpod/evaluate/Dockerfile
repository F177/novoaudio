FROM python:3.12-slim

WORKDIR /

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Modelo pré-baixado na imagem (roda em CPU no build, só pra baixar os pesos).
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu', compute_type='int8')"

COPY handler.py .
CMD ["python", "-u", "handler.py"]
