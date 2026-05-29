FROM python:3.12-slim

# ffmpeg is required: pipeline/audio.py shells out to it to decode browser audio.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# uv for fast, reproducible dependency installs.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# App source (pipeline/ is built as a package; static/ and app.py travel along).
COPY . .

# Install the project and its dependencies into the system interpreter.
RUN uv pip install --system --no-cache .

# app.py reads HOST/PORT from the environment; the host platform sets PORT.
ENV HOST=0.0.0.0
EXPOSE 8000

CMD ["python", "app.py"]
