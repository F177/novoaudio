"""Execução remota no Pod RunPod (SSH puro — sem serverless, sem R2).

Wrapper fino em cima de `ssh`/`scp` (OpenSSH, já usado manualmente durante
a validação desta fase — ver histórico da sessão). Não usa rsync porque a
máquina de desenvolvimento é Windows e rsync não é garantido; `scp -r` dá
conta do volume de arquivo que este pipeline move (poucos WAVs e JSONs por
vídeo, não milhares de arquivos pequenos).

Config lida de variáveis de ambiente (ver `.env.example`):
    POD_HOST, POD_PORT, POD_USER (default "root"), POD_SSH_KEY_PATH,
    POD_WORKSPACE (diretório no Pod onde o repo está espelhado, ex.
    "/workspace/novoaudio" — mesmo caminho usado nas sessões manuais de
    validação do T0.6-T0.9).
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass


@dataclass
class PodConfig:
    host: str
    port: int
    user: str
    key_path: str
    workspace: str

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
            key_path=os.environ["POD_SSH_KEY_PATH"],
            workspace=os.environ["POD_WORKSPACE"],
        )


def run_on_pod(config: PodConfig, command: str) -> None:
    """Roda `command` no Pod, dentro de `config.workspace` (via `cd &&`)."""
    subprocess.run(
        [
            "ssh",
            "-i",
            config.key_path,
            "-p",
            str(config.port),
            f"{config.user}@{config.host}",
            f"cd {config.workspace} && {command}",
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
