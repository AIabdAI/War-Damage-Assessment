# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base
WORKDIR /app

FROM base AS deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM deps AS runtime
COPY . .
CMD ["bash"]