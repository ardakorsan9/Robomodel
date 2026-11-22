# robomodel web app. Build:  docker build -t robomodel .
# Run:                       docker run -p 8000:8000 robomodel
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . && useradd --create-home robomodel
USER robomodel
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["robomodel", "serve", "--host", "0.0.0.0", "--port", "8000"]
