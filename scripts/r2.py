"""Cliente R2 mínimo pra buscar arquivos de teste do bucket, usado por
`scripts/dub.py`. Mesmo padrão de client S3-compatible dos
`runpod/*/handler.py`, mas rodando na máquina de desenvolvimento, não
num worker de GPU — por isso não vive dentro de `packages/pipeline`
(que não pode ter I/O de rede).
"""

from __future__ import annotations

import os
from pathlib import Path

import boto3


def _client():
    account_id = os.environ["R2_ACCOUNT_ID"]
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def download(key: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    _client().download_file(os.environ["R2_BUCKET"], key, str(local_path))
