"""Execução remota no Pod RunPod (SSH puro — sem serverless, sem R2).

Wrapper fino em cima de `ssh`/`scp` (OpenSSH, já usado manualmente durante
a validação desta fase — ver histórico da sessão). Não usa rsync porque a
máquina de desenvolvimento é Windows e rsync não é garantido; `scp -r` dá
conta do volume de arquivo que este pipeline move (poucos WAVs e JSONs por
vídeo, não milhares de arquivos pequenos).

Config lida de variáveis de ambiente (ver `.env.example`):
    POD_HOST, POD_PORT, POD_USER (default "root"), POD_SSH_KEY_PATH,
    POD_WORKSPACE (diretório no Pod onde o repo está espelhado — nesta
    validação, `/root/novoaudio_repo`, não `/workspace/...`: `/workspace`
    é o volume de rede, bom pra pesos de modelo grandes, ruim pra muita
    escrita de arquivo pequeno).

    POD_PYTHON_ASR / POD_PYTHON_TTS (opcionais): caminho do interpretador
    de cada venv já preparado no Pod. `separate_stems`/`transcribe`/
    `translate`/`evaluate` usam o venv "asr" (Demucs, WhisperX, pyannote,
    faster-whisper e também Qwen2.5 — `transformers` dessa venv já
    suporta a arquitetura `qwen2`, confirmado antes de rodar de verdade);
    `synthesize` usa o venv "tts" (MOSS-TTS, que pede um torch mais novo
    e incompatível com o resto). Sem essas variáveis, cai no caminho
    default anotado abaixo — mas se o Pod for recriado do zero, os
    venvs provavelmente vão morar em outro lugar, então ajuste o `.env`.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

_DEFAULT_PYTHON_ASR = "/root/venvs/asr/bin/python"
_DEFAULT_PYTHON_TTS = "/root/venvs/tts/bin/python"


@dataclass
class PodConfig:
    host: str
    port: int
    user: str
    key_path: str
    workspace: str
    python_asr: str
    python_tts: str

    @classmethod
    def from_env(cls) -> PodConfig:
        missing = [
            name
            for name in ("POD_HOST", "POD_PORT", "POD_SSH_KEY_PATH", "POD_WORKSPACE")
            if not os.environ.get(name)
        ]
        if missing:
            raise RuntimeError(
                f"variáveis de ambiente do Pod não configuradas: {', '.join(missing)}"
            )
        return cls(
            host=os.environ["POD_HOST"],
            port=int(os.environ["POD_PORT"]),
            user=os.environ.get("POD_USER", "root"),
            key_path=os.path.expanduser(os.environ["POD_SSH_KEY_PATH"]),
            workspace=os.environ["POD_WORKSPACE"],
            python_asr=os.environ.get("POD_PYTHON_ASR", _DEFAULT_PYTHON_ASR),
            python_tts=os.environ.get("POD_PYTHON_TTS", _DEFAULT_PYTHON_TTS),
        )


def run_on_pod(config: PodConfig, command: str, env: dict[str, str] | None = None) -> None:
    """Roda `command` no Pod, dentro de `config.workspace` (via `cd &&`).

    `env`, se passado, vira `KEY=VALUE` na frente do comando — necessário
    porque um comando SSH não-interativo não carrega `~/.bashrc` (onde
    `HF_HOME` normalmente estaria), confirmado na prática nesta sessão.
    """
    env_prefix = " ".join(f"{k}={v}" for k, v in (env or {}).items())
    full_command = f"{env_prefix} {command}".strip()
    subprocess.run(
        [
            "ssh",
            "-i",
            config.key_path,
            "-p",
            str(config.port),
            f"{config.user}@{config.host}",
            f"cd {config.workspace} && {full_command}",
        ],
        check=True,
    )


def upload_to_pod(config: PodConfig, local_path: str, remote_relative_path: str) -> None:
    subprocess.run(
        [
            "scp",
            "-i",
            config.key_path,
            "-P",
            str(config.port),
            "-r",
            local_path,
            f"{config.user}@{config.host}:{config.workspace}/{remote_relative_path}",
        ],
        check=True,
    )


def download_from_pod(config: PodConfig, remote_relative_path: str, local_path: str) -> None:
    subprocess.run(
        [
            "scp",
            "-i",
            config.key_path,
            "-P",
            str(config.port),
            "-r",
            f"{config.user}@{config.host}:{config.workspace}/{remote_relative_path}",
            local_path,
        ],
        check=True,
    )
