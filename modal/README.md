# Modal — wrappers de GPU

Quatro funções serverless (`separate_stems`, `transcribe`, `synthesize`,
`evaluate`) definidas em [`functions.py`](functions.py). Cada uma recebe e
devolve **chaves de objeto no R2**, nunca bytes de áudio direto na chamada.

## Setup (uma vez)

1. Conta no [modal.com](https://modal.com) e `modal setup` local (autentica o CLI).
2. Bucket no Cloudflare R2 e um par de chaves de acesso S3-compatível.
3. Token do Hugging Face com a licença de
   [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1)
   aceita na conta — sem isso a diarização em `transcribe` falha com 401/403.
4. Criar os dois secrets que `functions.py` referencia:

   ```bash
   modal secret create novoaudio-r2 \
     R2_ACCOUNT_ID=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... R2_BUCKET=...
   modal secret create novoaudio-hf HF_TOKEN=...
   ```

## Deploy

```bash
modal deploy modal/functions.py
```

O primeiro deploy baixa e "assa" os modelos nas imagens (Demucs, WhisperX
large-v3 + alinhamento + diarização, MOSS-TTS-v1.5 de 8B parâmetros, Whisper
large-v3 para avaliação) — pode levar bastante tempo e usa bastante disco de
build. Deploys seguintes reaproveitam essas camadas em cache enquanto as
versões pinadas não mudarem.

## Rodar contra um arquivo de teste

Suba um arquivo de teste para o bucket R2 primeiro (fora do escopo deste
módulo — via `aws s3 cp --endpoint-url ...` ou o console do Cloudflare), depois:

```bash
modal run modal/functions.py --stage separate_stems --key raw/teste.wav
modal run modal/functions.py --stage transcribe --key raw/teste.vocals.wav
modal run modal/functions.py --stage synthesize --text "Olá, mundo." --target-seconds 2.0
modal run modal/functions.py --stage evaluate --key test/synth.wav
```

Cada chamada imprime a chave R2 de saída (ou o dict de chaves, no caso de
`separate_stems`) e uma linha `[nome_da_função] modelo pronto em N.Ns` —
esse número é o proxy de cold start (tempo de carregar/mover o modelo para a
GPU; não inclui o tempo de start do container em si, que aparece no
dashboard do Modal ou em `modal app logs novoaudio-gpu`).

## Cold start medido

*(preencher depois de rodar de verdade — nenhuma medição foi feita nesta
sessão: não há conta Modal, GPU nem credenciais disponíveis aqui.)*

| Função | GPU | Cold start (container + modelo) | Warm (container já quente) |
|---|---|---|---|
| `separate_stems` | T4 | — | — |
| `transcribe` | A10G | — | — |
| `synthesize` | A10G | — | — |
| `evaluate` | T4 | — | — |

## Decisões e limitações desta versão

- **GPU por função** é um ponto de partida (T4 para Demucs/eval, A10G para
  WhisperX e MOSS-TTS), não medido. Ajustar depois de rodar de verdade —
  se `synthesize` estourar VRAM em A10G (24 GB, modelo de 8B em bf16 já usa
  uns 16 GB só de pesos), subir para A100.
- **FlashAttention 2 desligado por padrão** na imagem do MOSS-TTS: compilar
  do zero deixa o build frágil e lento. Para habilitar, trocar o
  `pip install -e .` por `pip install -e ".[flash-attn]"` em `functions.py`
  e confirmar que a GPU tem compute capability ≥8 (Ampere ou mais nova).
- **`evaluate` só transcreve.** O cálculo de CER (comparar a transcrição com
  o texto alvo) é lógica pura de CPU e vive em `packages/pipeline/quality.py`
  (T0.11), não em GPU.
- Nenhuma função, imagem ou modelo foi executado nesta sessão. O código foi
  escrito contra a documentação e os model cards reais (model card do
  MOSS-TTS-v1.5 no Hugging Face, READMEs do WhisperX/Demucs/faster-whisper),
  e a assinatura das APIs do Modal (`Image`, `App.function`, `Secret`) foi
  conferida por introspecção contra o pacote `modal` instalado localmente —
  mas isso não substitui rodar de verdade. Trate o critério de aceite desta
  tarefa como pendente até a tabela acima estar preenchida.
