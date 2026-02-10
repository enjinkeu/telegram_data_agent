# Use a slim Python image
FROM python:3.10-slim

# Install OpenJDK-17 (Required for PySpark 3.x)
RUN apt-get update && \
    apt-get install -y openjdk-17-jre-headless procps && \
    apt-get clean;

# Set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Install Poetry
RUN pip install poetry

WORKDIR /app

# Copy dependency definition
COPY pyproject.toml poetry.lock* /app/

# Install dependencies (no dev dependencies for prod, no interaction)
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Copy source code
COPY src /app/src
COPY configs /app/configs

# Create data directory for mounting volumes
RUN mkdir -p /app/data

# Entry point
CMD ["python", "-m", "src.pipelines.telegram_import"]