# Camada fina de cima da imagem base (build/push só quando dependências ou
# o modelo mudam — ver base.Dockerfile). Mexer só no handler.py? Rebuild
# disso aqui é questão de segundos, não minutos, porque reaproveita a base
# já pronta local ou puxada do Hub.
# Build context é a raiz do repo (o handler importa packages/pipeline):
#   docker buildx build -f runpod/synthesize/Dockerfile .
FROM f177/novoaudio-synthesize-base:latest

WORKDIR /

COPY runpod/synthesize/handler.py .
CMD ["python", "-u", "handler.py"]
