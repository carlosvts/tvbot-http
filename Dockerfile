# ─────────────────────────────────────────────────────────────────────────────
# Imagem do CEREBRO (server).
#
# Base com CUDA 12 + cuDNN 9: o faster-whisper (CTranslate2) precisa das libs
# de cuDNN em runtime, e a imagem "runtime" da NVIDIA já traz tudo.
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# python3.10 é o padrão do Ubuntu 22.04 e cobre todas as dependências.
# build-essential é necessário para compilar o webrtcvad (extensão C).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        build-essential \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir -r requirements.txt

COPY server/ /app/

# Volume dos modelos (wake word customizada, voz do Piper, cache do Whisper).
VOLUME ["/models"]
# Cache do Whisper (baixado pelo faster-whisper na primeira execução).
ENV HF_HOME=/models/hf

EXPOSE 9876

# Baixa o que faltar e só então sobe o server.
CMD ["sh", "-c", "python3 download_models.py && python3 main.py"]