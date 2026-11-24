FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-space.txt .
RUN pip install --no-cache-dir "torch>=2.0" --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-space.txt

COPY main.py reference_corpus.jsonl entrypoint.sh ./
RUN chmod +x entrypoint.sh

EXPOSE 7860
CMD ["./entrypoint.sh"]
