# ──────────────────────────────────────────────────────────────────────────────
# EchoSeal – realtime audio watermarking
# Build:
#   docker build -t echoseal:latest .
#
# Run live TX (Linux + ALSA/Pulse):
#   docker run --rm -it --device /dev/snd echoseal:latest \
#          echoseal-tx --key $(openssl rand -hex 32)
#
# Verify a file (no audio device needed):
#   docker run --rm -v $PWD:/data echoseal \
#          echoseal-rx --key <key> /data/recording.wav
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim

# 1) basic tools & build deps for numpy / scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git libsndfile1-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

# 2) copy source & install
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir .[dev]      # prod: just "."
# strip testing deps by changing to "."
# optional:   && python -m pytest -q

# 3) create unprivileged user
RUN useradd -m echoseal
USER echoseal

# 4) default cmd shows help
ENTRYPOINT ["echoseal-tx"]
CMD ["--help"]
