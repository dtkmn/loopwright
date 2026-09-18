# Use a current Python runtime with security updates.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /uvx /usr/local/bin/

# Set the working directory
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PYTHON_DOWNLOADS=0 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Cache third-party dependencies separately from application source.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# Copy the rest of the application code into the image
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# Create a non-root user and set permissions
RUN addgroup --gid 1000 appuser && \
    adduser --uid 1000 --gid 1000 --disabled-password --gecos "" appuser && \
    mkdir -p /app/uploads /home/appuser/.cache && \
    chown -R appuser:appuser /app /home/appuser/.cache

# Switch to the non-root user
USER appuser

# Expose the port the app runs on
EXPOSE 7860

# Set environment variables for the application
ENV WEB_HOST="0.0.0.0"
ENV WEB_PORT="7860"
ENV FAST_MODE="true"
ENV LLM_BACKEND="auto"
ENV APP_DEBUG="false"

# Run the installed entrypoint without resolving dependencies at startup.
CMD ["loopwright"]
