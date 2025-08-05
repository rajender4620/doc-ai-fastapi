# 📝 FastAPI AI Document Assistant

A simple FastAPI app that:
- accepts PDF/DOCX uploads
- extracts and redacts PII using spaCy
- summarizes and detects compliance risks using OpenAI (via OpenRouter)
- stores summaries in Chroma vector DB
- provides semantic search over uploaded docs

## 🚀 Endpoints
- `POST /upload` — upload file, get redacted text, summary & risks
- `GET /search?q=...` — search stored summaries
- `GET /health` — health check

## 🛠 Local dev
```bash
pip install -r requirements.txt
uvicorn main:app --reload
