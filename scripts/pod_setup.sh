#!/usr/bin/env bash
# Monta as duas venvs do harness de GPU da Fase 0 num Pod novo (A40, sem MIG).
# Roda NO Pod, depois de subir o código (packages/, scripts/, pyproject.toml,
# voices/). Versões fixadas pelos rebuilds reais de 2026-10-05/09: whisperx
# atual exige torch>=2.8 (via pyannote-audio>=4.0.7); MOSS-TTS-v1.5 exige
# transformers==5.0.0 exato; torchcodec 0.8.1 do índice cu128.
set -euo pipefail

CU128=https://download.pytorch.org/whl/cu128
mkdir -p /root/novoaudio_repo/voices /root/hf_cache /root/venvs

setup_asr() {
  python3.11 -m venv /root/venvs/asr
  local pip=/root/venvs/asr/bin/pip
  $pip install -q torch==2.8.0 torchaudio==2.8.0 --index-url $CU128
  $pip install -q whisperx demucs transformers accelerate soundfile funasr modelscope faster-whisper
  # whisperx/funasr sobem o torch por baixo dos panos — refixa.
  $pip install -q torch==2.8.0 torchaudio==2.8.0 --index-url $CU128
  $pip check
  echo ASR_OK
}

setup_tts() {
  python3.11 -m venv /root/venvs/tts
  local pip=/root/venvs/tts/bin/pip
  $pip install -q torch==2.9.1 torchaudio==2.9.1 --index-url $CU128
  $pip install -q torchcodec==0.8.1 --index-url $CU128
  $pip install -q transformers==5.0.0 accelerate soundfile librosa peft
  echo TTS_OK
}

setup_asr > /root/setup_asr.log 2>&1 &
setup_tts > /root/setup_tts.log 2>&1 &
wait
tail -n 1 /root/setup_asr.log /root/setup_tts.log
/root/venvs/asr/bin/python -c "import torch; print('asr torch', torch.__version__, torch.cuda.is_available())"
/root/venvs/tts/bin/python -c "import torch, transformers; print('tts torch', torch.__version__, 'transformers', transformers.__version__)"
