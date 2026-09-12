"""Shim de compatibilidade: pyannote.audio==3.1.1 (pinado transitivamente
pelo whisperx 3.2.0) usa a API antiga de backend do torchaudio
(get/set/list_audio_backend), removida nas versões novas do torchaudio.
Isso é incompatibilidade do upstream, não nossa — importar este módulo
antes de qualquer `import whisperx`.
"""

import torchaudio

if not hasattr(torchaudio, "get_audio_backend"):
    torchaudio.get_audio_backend = lambda: "soundfile"
if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda *args, **kwargs: None
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]
