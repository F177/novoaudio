"""Entrypoints de GPU (Modal) do pipeline de dublagem.

Quatro funções serverless, cada uma recebe e devolve **chaves de objeto no
R2** (nunca bytes de áudio na assinatura) — a orquestração em `apps/api`
decide o que fazer com o resultado:

- `separate_stems`  (Demucs/htdemucs)   -> {"vocals": key, "background": key}
- `transcribe`      (WhisperX + pyannote) -> key do JSON com segments/words/speakers
- `synthesize`      (MOSS-TTS-v1.5)     -> key do wav sintetizado
- `evaluate`        (Whisper large-v3)  -> key do JSON com a transcrição (CER
                                            round-trip é calculado em
                                            packages/pipeline/quality.py, que
                                            não roda em GPU)

Credenciais via Modal Secrets (criar antes do primeiro deploy):

    modal secret create novoaudio-r2 \\
        R2_ACCOUNT_ID=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... R2_BUCKET=...
    modal secret create novoaudio-hf HF_TOKEN=...

`HF_TOKEN` precisa ter aceitado a licença de
`pyannote/speaker-diarization-3.1` no Hugging Face, senão a diarização falha.

GPU escolhida por função (ajustar depois de medir custo/latência real —
ver `modal/README.md`):

- `separate_stems`: T4 (Demucs é leve).
- `transcribe`: A10G (WhisperX large-v3 + diarização).
- `synthesize`: A10G (MOSS-TTS-v1.5 tem 8B parâmetros em bf16, ~16 GB só de
  pesos; sem FlashAttention 2 por padrão — ver nota na imagem).
- `evaluate`: T4 (só transcrição, sem diarização).

Limitações conhecidas / não verificado nesta sessão: não há conta Modal,
bucket R2 ou GPU disponíveis aqui para rodar de verdade. O código foi escrito
contra a documentação e o model card reais (MOSS-TTS-v1.5, WhisperX 3.2,
Demucs 4.1, faster-whisper 1.2), mas o critério de aceite desta tarefa
("cada função roda num arquivo de teste... cold start medido") só se cumpre
quando alguém rodar isso de verdade — ver `modal/README.md`.
"""

from __future__ import annotations

import modal

app = modal.App("novoaudio-gpu")

r2_secret = modal.Secret.from_name("novoaudio-r2")
hf_secret = modal.Secret.from_name("novoaudio-hf")


# --------------------------------------------------------------------------
# R2 (as chamadas de rede só acontecem dentro dos containers do Modal, por
# isso os imports ficam dentro das funções em vez de no topo do módulo)
# --------------------------------------------------------------------------


def _r2_client():
    import os

    import boto3

    account_id = os.environ["R2_ACCOUNT_ID"]
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def _download_from_r2(key: str) -> str:
    import os
    import tempfile

    local_path = os.path.join(tempfile.mkdtemp(), os.path.basename(key))
    _r2_client().download_file(os.environ["R2_BUCKET"], key, local_path)
    return local_path


def _upload_to_r2(local_path: str, key: str) -> str:
    import os

    _r2_client().upload_file(local_path, os.environ["R2_BUCKET"], key)
    return key


# --------------------------------------------------------------------------
# Imagens — uma por função, cada uma com o modelo pré-baixado (build-time)
# --------------------------------------------------------------------------


def _bake_demucs() -> None:
    import demucs.api

    demucs.api.Separator(model="htdemucs")


demucs_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("demucs==4.1.0", "boto3")
    .run_function(_bake_demucs)
)


def _bake_whisperx() -> None:
    import os

    import whisperx
    from whisperx.diarize import DiarizationPipeline

    whisperx.load_model("large-v3", "cpu", compute_type="int8")
    whisperx.load_align_model(language_code="en", device="cpu")
    DiarizationPipeline(token=os.environ["HF_TOKEN"], device="cpu")


whisperx_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install("whisperx==3.2.0", "boto3")
    .run_function(_bake_whisperx, secrets=[hf_secret])
)


def _bake_moss_tts() -> None:
    from transformers import AutoModel, AutoProcessor

    repo_id = "OpenMOSS-Team/MOSS-TTS-v1.5"
    AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)
    AutoModel.from_pretrained(repo_id, trust_remote_code=True)


moss_tts_image = (
    modal.Image.debian_slim(python_version="3.12")
    # FlashAttention 2 fica de fora por padrão: exige compilar do zero contra
    # uma GPU com compute capability >=8 (Ampere+) e deixa o build frágil.
    # Para habilitar: trocar o `pip install -e .` abaixo por
    # `pip install -e ".[flash-attn]"` e usar gpu="A10G" ou melhor.
    .apt_install("git", "ffmpeg", "build-essential")
    .pip_install(
        "torch==2.9.1", "torchaudio==2.9.1",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/OpenMOSS/MOSS-TTS.git /root/moss-tts",
        "cd /root/moss-tts && pip install --extra-index-url "
        "https://download.pytorch.org/whl/cu128 -e .",
    )
    .pip_install("boto3")
    .run_function(_bake_moss_tts)
)


def _bake_eval_whisper() -> None:
    from faster_whisper import WhisperModel

    WhisperModel("large-v3", device="cpu", compute_type="int8")


eval_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("faster-whisper==1.2.1", "boto3")
    .run_function(_bake_eval_whisper)
)


# --------------------------------------------------------------------------
# Funções
# --------------------------------------------------------------------------


@app.function(
    image=demucs_image,
    gpu="T4",
    secrets=[r2_secret],
    timeout=600,
    scaledown_window=180,
)
def separate_stems(source_key: str) -> dict[str, str]:
    """Separa voz e fundo com Demucs (htdemucs)."""
    import os
    import time

    import demucs.api

    t0 = time.monotonic()
    local_source = _download_from_r2(source_key)
    separator = demucs.api.Separator(model="htdemucs", device="cuda")
    _original, separated = separator.separate_audio_file(local_source)
    print(f"[separate_stems] modelo pronto em {time.monotonic() - t0:.1f}s")

    vocals = separated["vocals"]
    background = separated["drums"] + separated["bass"] + separated["other"]

    stem, _ext = os.path.splitext(source_key)
    vocals_path, background_path = "/tmp/vocals.wav", "/tmp/background.wav"
    demucs.api.save_audio(vocals, vocals_path, samplerate=separator.samplerate)
    demucs.api.save_audio(background, background_path, samplerate=separator.samplerate)

    vocals_key = f"{stem}.vocals.wav"
    background_key = f"{stem}.background.wav"
    _upload_to_r2(vocals_path, vocals_key)
    _upload_to_r2(background_path, background_key)
    return {"vocals": vocals_key, "background": background_key}


@app.function(
    image=whisperx_image,
    gpu="A10G",
    secrets=[r2_secret, hf_secret],
    timeout=900,
    scaledown_window=180,
)
def transcribe(audio_key: str, language: str = "en") -> str:
    """ASR com timestamps por palavra e diarização (WhisperX + pyannote)."""
    import json
    import os
    import time

    import whisperx
    from whisperx.diarize import DiarizationPipeline

    device = "cuda"
    t0 = time.monotonic()
    local_audio = _download_from_r2(audio_key)

    model = whisperx.load_model("large-v3", device, compute_type="float16", language=language)
    print(f"[transcribe] modelo ASR pronto em {time.monotonic() - t0:.1f}s")

    audio = whisperx.load_audio(local_audio)
    result = model.transcribe(audio, batch_size=16)

    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
    result = whisperx.align(
        result["segments"], model_a, metadata, audio, device, return_char_alignments=False
    )

    diarize_model = DiarizationPipeline(token=os.environ["HF_TOKEN"], device=device)
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    output_key = f"{os.path.splitext(audio_key)[0]}.transcript.json"
    local_output = "/tmp/transcript.json"
    with open(local_output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    _upload_to_r2(local_output, output_key)
    return output_key


@app.function(
    image=moss_tts_image,
    gpu="A10G",
    secrets=[r2_secret, hf_secret],
    timeout=300,
    scaledown_window=300,
)
def synthesize(
    text: str,
    target_seconds: float,
    output_key: str,
    reference_audio_key: str | None = None,
    language: str = "Portuguese",
) -> str:
    """Sintetiza `text` com MOSS-TTS mirando `target_seconds` de duração.

    `text` já deve trazer os marcadores `[pause X.Ys]` e IPA (`/.../`) prontos
    — isso é responsabilidade de `packages/pipeline/synthesis.py` (T0.9), não
    desta função. `language="Portuguese"` porque o modelo tende ao pt-PT sem
    essa tag explícita (ver `docs/ptbr.md`).
    """
    import time

    import torch
    import torchaudio
    from transformers import AutoModel, AutoProcessor

    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    repo_id = "OpenMOSS-Team/MOSS-TTS-v1.5"
    device = "cuda"

    t0 = time.monotonic()
    processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)
    processor.audio_tokenizer = processor.audio_tokenizer.to(device)
    model = AutoModel.from_pretrained(
        repo_id, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    print(f"[synthesize] modelo pronto em {time.monotonic() - t0:.1f}s")

    tokens = round(target_seconds * 12.5)  # 12.5 Hz, granularidade de 80ms
    reference = [_download_from_r2(reference_audio_key)] if reference_audio_key else None

    message = processor.build_user_message(
        text=text, language=language, tokens=tokens, reference=reference
    )
    batch = processor([[message]], mode="generation")
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=4096
    )

    local_output = "/tmp/synth.wav"
    for decoded in processor.decode(outputs):
        audio = decoded.audio_codes_list[0]
        torchaudio.save(local_output, audio.unsqueeze(0), processor.model_config.sampling_rate)
        break

    _upload_to_r2(local_output, output_key)
    return output_key


@app.function(
    image=eval_image,
    gpu="T4",
    secrets=[r2_secret],
    timeout=300,
    scaledown_window=180,
)
def evaluate(audio_key: str, language: str = "pt") -> str:
    """Transcreve `audio_key` com Whisper large-v3 — insumo para o CER round-trip.

    Só devolve a transcrição; comparar com o texto alvo é trabalho de CPU e
    fica em `packages/pipeline/quality.py` (T0.11), não aqui.
    """
    import json
    import os
    import time

    from faster_whisper import WhisperModel

    t0 = time.monotonic()
    local_audio = _download_from_r2(audio_key)
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    print(f"[evaluate] modelo pronto em {time.monotonic() - t0:.1f}s")

    segments, _info = model.transcribe(local_audio, language=language, beam_size=5)
    text = " ".join(segment.text.strip() for segment in segments)

    output_key = f"{os.path.splitext(audio_key)[0]}.eval.json"
    local_output = "/tmp/eval.json"
    with open(local_output, "w", encoding="utf-8") as f:
        json.dump({"text": text}, f, ensure_ascii=False)
    _upload_to_r2(local_output, output_key)
    return output_key


@app.local_entrypoint()
def main(
    stage: str,
    key: str = "",
    text: str = "",
    target_seconds: float = 3.0,
    output_key: str = "test/synth.wav",
) -> None:
    """Smoke test manual — ver `modal/README.md` para exemplos completos.

    `modal run modal/functions.py --stage separate_stems --key raw/test.wav`
    """
    if stage == "separate_stems":
        print(separate_stems.remote(key))
    elif stage == "transcribe":
        print(transcribe.remote(key))
    elif stage == "synthesize":
        print(synthesize.remote(text=text, target_seconds=target_seconds, output_key=output_key))
    elif stage == "evaluate":
        print(evaluate.remote(key))
    else:
        raise ValueError(f"stage desconhecido: {stage!r}")
