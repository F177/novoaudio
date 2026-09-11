FROM python:3.12-slim

WORKDIR /

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Modelo pré-baixado na imagem (htdemucs, leve, roda em CPU no build).
RUN python -c "import demucs.api; demucs.api.Separator(model='htdemucs')"

COPY handler.py .
CMD ["python", "-u", "handler.py"]
