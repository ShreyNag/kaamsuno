# KaamSuno voice incident reporter. Build and run with: docker compose up --build
# Same Python as the dev venv (3.12.9); CPU only (onnxruntime, no PyTorch).
FROM python:3.12.9-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1

# libsndfile1: soundfile; libgomp1: onnxruntime's OpenMP. PyAV (aiortc) ships its own FFmpeg libraries.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libsndfile1 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.lock.txt .
RUN pip install -r requirements.lock.txt

# The Silero VAD and Smart Turn v3 ONNX models ship inside the pipecat wheel. Load both once here, so a
# missing or broken model fails the build, not the first call, and nothing is fetched at runtime.
RUN python -c "from pipecat.audio.vad.silero import SileroVADAnalyzer as S; S(); \
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3 as T; T(); \
print('models OK')"

COPY app ./app
COPY research ./research
COPY tests/fixtures ./tests/fixtures

RUN useradd --create-home --uid 1000 kaamsuno \
 && mkdir -p logs reports \
 && chown -R kaamsuno:kaamsuno /app
USER kaamsuno

EXPOSE 7860
# --host 0.0.0.0: the runner binds localhost by default, which is unreachable from outside the container
CMD ["python", "-m", "app.bot", "-t", "webrtc", "--host", "0.0.0.0", "--port", "7860"]
