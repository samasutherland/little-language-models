FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace
WORKDIR /workspace

COPY . .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["bash","-lc","timeout ${TRAIN_MINS:-30}m python3 train_model.py"]