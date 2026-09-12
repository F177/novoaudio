# RunPod Serverless — wrappers de GPU

Quatro workers, cada um seu próprio endpoint serverless e sua própria imagem
Docker: [`separate_stems/`](separate_stems), [`transcribe/`](transcribe),
[`synthesize/`](synthesize), [`evaluate/`](evaluate). Cada handler recebe e
devolve **chaves de objeto no R2**, nunca bytes de áudio na chamada.

## Setup (uma vez)

1. Conta no [RunPod](https://runpod.io) e uma API key (Settings → API Keys).
2. Conta no Docker Hub (`docker login`).
3. Bucket no Cloudflare R2 e um par de chaves de acesso S3-compatível.
4. Token do Hugging Face com a licença de
   [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1)
   aceita na conta — sem isso o build de `transcribe/` falha.

## Build e push das imagens

RunPod exige `linux/amd64`. Rodar da raiz do repo.

`transcribe` e `synthesize` são pesados (dependências frágeis, modelo de
8B) e mudam pouco depois de prontos — por isso ficam em **duas imagens**:
uma `base.Dockerfile` (deps + modelo, builda uma vez, raro mudar) e um
`Dockerfile` fino em cima (`FROM <base>` + `COPY handler.py`). Mexer só na
lógica do handler? Só reconstrói/repush a imagem fina — segundos, não
15-20 minutos. Só refaça a base se mudar dependência/versão de modelo.

```bash
# --- bases (raro rodar de novo) ---
docker buildx build --platform linux/amd64 \
  -f runpod/transcribe/base.Dockerfile \
  --secret id=hf_token,env=HF_TOKEN \
  -t <seu-usuario>/novoaudio-transcribe-base:latest \
  runpod/transcribe --push

docker buildx build --platform linux/amd64 \
  -f runpod/synthesize/base.Dockerfile \
  -t <seu-usuario>/novoaudio-synthesize-base:latest \
  runpod/synthesize --push

# --- imagens finais (essas sim mudam sempre que mexer no handler.py) ---
docker buildx build --platform linux/amd64 \
  -t <seu-usuario>/novoaudio-separate-stems:latest \
  runpod/separate_stems --push

docker buildx build --platform linux/amd64 \
  -t <seu-usuario>/novoaudio-transcribe:latest \
  runpod/transcribe --push

docker buildx build --platform linux/amd64 \
  -t <seu-usuario>/novoaudio-synthesize:latest \
  runpod/synthesize --push

docker buildx build --platform linux/amd64 \
  -t <seu-usuario>/novoaudio-evaluate:latest \
  runpod/evaluate --push
```

`transcribe` precisa do `HF_TOKEN` só no build da imagem **base** (baixa o
modelo de diarização); a imagem final e os outros três workers não.

**Nota:** o `FROM` das imagens finais de `transcribe`/`synthesize` está
com o usuário `f177` do Docker Hub fixo (é o já usado neste projeto) — se
trocar de conta, ajustar o `FROM` nos dois `Dockerfile` antes de buildar.

As imagens de `synthesize` (MOSS-TTS, 8B parâmetros) e `transcribe`
(WhisperX large-v3 + diarização) ficam grandes — o build demora e usa
bastante disco. Isso é esperado.

## Criar os endpoints

No console do RunPod, criar 4 endpoints serverless, um por imagem:

| Endpoint | Imagem | GPU sugerida | Por quê |
|---|---|---|---|
| `separate_stems` | `novoaudio-separate-stems` | 16 GB (ex. RTX 4000 Ada / A4000) — o mais barato | Demucs/htdemucs é pequeno (~150MB) |
| `transcribe` | `novoaudio-transcribe` | 24 GB (RTX 3090/4090, A5000) | large-v3 + alinhamento + diarização carregados juntos |
| `synthesize` | `novoaudio-synthesize` | 48 GB (A6000, L40, RTX 6000 Ada) | 8B parâmetros em bf16 (~16 GB só de pesos); sem FlashAttention 2, o cache de atenção na geração consome mais — 24 GB é arriscado de estourar |
| `evaluate` | `novoaudio-evaluate` | 16 GB | Whisper large-v3 sozinho em float16 cabe folgado |

Estimativa por tamanho de modelo, não medida — se topar rodar `synthesize`
em 24 GB pra economizar e não estourar OOM, fique à vontade; só espere que
possa quebrar.

Em cada endpoint, configurar como variáveis de ambiente: `R2_ACCOUNT_ID`,
`R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET` (todos) e `HF_TOKEN`
(só no `transcribe`, em runtime — a diarização usa o token de novo, não só no
build).

Guardar os 4 IDs de endpoint em `.env` como `RUNPOD_ENDPOINT_*` (ver
`.env.example`).

## Rodar contra um arquivo de teste

Suba um arquivo de teste para o bucket R2 primeiro (fora do escopo deste
módulo), depois chame o endpoint via `runsync`:

```bash
curl -X POST "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_SEPARATE_STEMS/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"source_key": "raw/teste.wav"}}'

curl -X POST "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_TRANSCRIBE/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"audio_key": "raw/teste.vocals.wav"}}'

curl -X POST "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_SYNTHESIZE/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"text": "Olá, mundo.", "target_seconds": 2.0, "output_key": "test/synth.wav"}}'

curl -X POST "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_EVALUATE/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"audio_key": "test/synth.wav"}}'
```

A resposta JSON traz `output` com a chave R2 do resultado (e `executionTime`
em ms — comparar a primeira chamada, que sofre cold start, com uma segunda
logo em seguida no worker já quente, para medir o cold start real).

## Cold start medido

*(preencher depois de rodar de verdade — nenhuma medição foi feita nesta
sessão: não há conta RunPod, Docker Hub, GPU nem credenciais disponíveis
aqui.)*

| Worker | GPU | Cold start (`executionTime` na 1ª chamada) | Warm (chamada seguinte) |
|---|---|---|---|
| `separate_stems` | | — | — |
| `transcribe` | | — | — |
| `synthesize` | | — | — |
| `evaluate` | | — | — |

## Decisões e limitações desta versão

- **GPU por endpoint** é escolhida no console do RunPod, não no código —
  a tabela acima é ponto de partida, ajustar depois de rodar de verdade.
- **Modelo pré-baixado na imagem** para os quatro workers (em vez do recurso
  de "cached models" do RunPod, que baixa via rede na primeira chamada) —
  mais simples de raciocinar, ao custo de imagens maiores e builds mais
  lentos. Migrar pra cached models depois é uma otimização de custo, não
  uma mudança de arquitetura.
- **FlashAttention 2 desligado por padrão** no `synthesize`: compilar do
  zero deixa o build frágil. Ver comentário no `Dockerfile` pra habilitar.
- **`evaluate` só transcreve.** O cálculo de CER (comparar a transcrição com
  o texto alvo) é lógica pura de CPU e vive em `packages/pipeline/quality.py`
  (T0.11), não em GPU.
- Cada `handler.py` duplica a mesma dúzia de linhas de I/O com R2 em vez de
  importar um módulo compartilhado — proposital: cada worker é seu próprio
  contexto de build Docker isolado, e a duplicação é mais simples que
  resolver isso com um pacote local instalável.
- Nenhuma imagem, endpoint ou modelo foi testado nesta sessão. O código foi
  escrito contra a documentação real (model card do MOSS-TTS-v1.5 no
  Hugging Face, docs do RunPod Serverless, READMEs do WhisperX/Demucs/
  faster-whisper) — mas isso não substitui rodar de verdade. Trate o
  critério de aceite desta tarefa como pendente até a tabela acima estar
  preenchida.
