FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace
WORKDIR /workspace
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates unzip \
 && curl -fsSL https://rclone.org/install.sh | bash \
 && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir -r requirements.txt
RUN chmod +x scripts/start.sh
CMD ["bash","-lc","./scripts/start.sh"]