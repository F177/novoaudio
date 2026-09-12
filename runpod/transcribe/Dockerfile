# Camada fina de cima da imagem base (build/push só quando dependências ou
# modelos mudam — ver base.Dockerfile). Mexer só no handler.py? Rebuild
# disso aqui é questão de segundos, não minutos, porque reaproveita a base
# já pronta local ou puxada do Hub.
FROM f177/novoaudio-transcribe-base:latest

WORKDIR /

COPY handler.py .
CMD ["python", "-u", "handler.py"]
