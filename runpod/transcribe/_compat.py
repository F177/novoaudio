"""Shim de compatibilidade: pyannote.audio==3.1.1 (pinado transitivamente
pelo whisperx 3.2.0) usa a API antiga de backend do torchaudio
(get/set/list_audio_backend), removida nas versões novas do torchaudio.
Isso é incompatibilidade do upstream, não nossa — importar este módulo
antes de qualquer `import whisperx`.

Achado real 2026-09-25 (Pod com GPU Blackwell, exige torchaudio 2.9.1+ pra
ter kernel sm_120): essa versão também removeu o MÓDULO `torchaudio.backend`
inteiro e a classe `torchaudio.AudioMetaData` (não só moveu — pyannote usa
`torchaudio.AudioMetaData` como type hint avaliado na hora do import, e
`from torchaudio.backend.common import AudioMetaData` em outro lugar).
Mesmo shim de `scripts/pod_worker.py::_patch_torch_compat_shims`,
mantido igual aqui pra não divergir.
"""

import dataclasses
import sys
import types

import torchaudio

if not hasattr(torchaudio, "get_audio_backend"):
    torchaudio.get_audio_backend = lambda: "soundfile"
if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda *args, **kwargs: None
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

if not hasattr(torchaudio, "AudioMetaData"):

    @dataclasses.dataclass
    class AudioMetaData:
        sample_rate: int
        num_frames: int
        num_channels: int
        bits_per_sample: int = 16
        encoding: str = "PCM_S"

    torchaudio.AudioMetaData = AudioMetaData

if not hasattr(torchaudio, "backend"):
    backend_module = types.ModuleType("torchaudio.backend")
    common_module = types.ModuleType("torchaudio.backend.common")
    common_module.AudioMetaData = torchaudio.AudioMetaData
    backend_module.common = common_module
    torchaudio.backend = backend_module
    sys.modules["torchaudio.backend"] = backend_module
    sys.modules["torchaudio.backend.common"] = common_module

# Achado real 2026-09-25, mesmo Pod: torch 2.6+ mudou o default de
# `torch.load` pra `weights_only=True` (segurança). O checkpoint do
# pyannote (carregado via `lightning_fabric.utilities.cloud_io`) tem
# objetos customizados (`torch.torch_version.TorchVersion`,
# `pyannote.audio.core.task.Specifications`, e provavelmente mais) que não
# estão na lista segura por padrão, e `lightning_fabric` passa
# `weights_only=True` explicitamente (só dar allowlist com
# `add_safe_globals` não bastou). Força `weights_only=False` pra qualquer
# `torch.load` depois deste ponto — aceitável aqui porque são checkpoints
# oficiais do HuggingFace (pyannote, whisperx), não arquivo de origem
# desconhecida.
import functools

import torch

if not getattr(torch.load, "_novoaudio_patched", False):
    _original_torch_load = torch.load

    @functools.wraps(_original_torch_load)
    def _torch_load_trusted_default(*args, **kwargs):
        kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)

    _torch_load_trusted_default._novoaudio_patched = True
    torch.load = _torch_load_trusted_default
