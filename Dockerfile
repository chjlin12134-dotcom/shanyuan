FROM python:3.11-slim

WORKDIR /app

COPY requirements-render.txt .
RUN pip install --no-cache-dir -r requirements-render.txt

COPY app_fastapi.py .
COPY index.html .
COPY system_prompt.md .
COPY shanyuan_corpus.csv .

EXPOSE 7860

CMD ["uvicorn", "app_fastapi:app", "--host", "0.0.0.0", "--port", "7860"]
