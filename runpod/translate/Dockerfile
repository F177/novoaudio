# Camada fina de cima da imagem base (build/push só quando dependências ou
# o modelo mudam — ver base.Dockerfile). Mexer só no handler.py? Rebuild
# disso aqui é questão de segundos, não minutos.
# Build context ainda é a raiz do repo (por causa do handler.py, que é o
# único arquivo copiado aqui — mas mantemos consistência com o base):
#   docker buildx build -f runpod/translate/Dockerfile .
FROM f177/novoaudio-translate-base:latest

WORKDIR /

COPY runpod/translate/handler.py .
CMD ["python", "-u", "handler.py"]
