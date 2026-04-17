# Mini Agentic QA Service

A small production-minded Flask + Ollama service that chooses tools, executes them, and returns grounded structured JSON responses with citations when available.

## What this project does

This service answers user questions by selecting and calling tools at runtime.

It currently supports:

- `web_search` using the DuckDuckGo Instant Answer API
- `get_weather` using a small local dummy weather knowledge base backed by a FAISS index

The app returns a structured JSON response in this shape:

```json
{
  "answer": "string",
  "sources": [{"name": "string", "url": "string"}],
  "latency_ms": {
    "total": 123,
    "by_step": {
      "retrieve": 45,
      "llm": 60
    }
  },
  "tokens": {
    "prompt": 0,
    "completion": 0
  },
  "tool_trace": [
    {
      "tool": "web_search",
      "args": {"query": "example"},
      "latency_ms": 10,
      "cache_hit": false,
      "ok": true
    }
  ]
}
```

## Main features

- Plain Python planner + tool registry
- Two tools with clear input schemas
- Structured final JSON output
- JSON Schema validation for tool inputs and final output
- Retries and timeouts for HTTP requests
- TTL cache for repeated tool calls
- Basic concurrency limits for tool execution
- Streaming endpoint using Server-Sent Events
- Simple URL safety filtering for public sources only
- Lightweight weather retrieval using a locally built FAISS index
- Unit tests for key utility and endpoint behavior

## Project files

- `app.py` — main Flask application, planner, tool registry, and endpoints
- `faiss_db.py` — builds the local FAISS weather index and document store
- `test_app.py` — basic tests
- `requirements.txt` — Python dependencies
- `Dockerfile` — container setup
- `.env` — local environment variables

## Requirements

- Python 3.10+
- Ollama installed and running locally
- A pulled Ollama model, default: `llama3.1:8b`

## Environment variables

Create a `.env` file in the project root with:

```env
OLLAMA_URL=http://localhost:11434/api/chat
OLLAMA_MODEL=llama3.1:8b
HOST=0.0.0.0
PORT=5000
```

### Variable details

- `OLLAMA_URL` — Ollama chat endpoint
- `OLLAMA_MODEL` — model name used by the app
- `HOST` — Flask host
- `PORT` — Flask port

## Local setup

### 1. Create and activate a virtual environment

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Ollama and pull the model

```bash
ollama pull llama3.1:8b
ollama serve
```

### 4. Build the local weather FAISS database

This step is required before running the app because the weather tool reads from the persisted index files created here.

```bash
python faiss_db.py
```

This creates:

- `weather_store/weather.index`
- `weather_store/weather_docs.json`

### 5. Run the app

```bash
python app.py
```

The app will start on:

- `http://0.0.0.0:5000`
- Open `http://localhost:5000` in your browser

## Running tests

### Run the built-in self-test

```bash
python app.py --self-test
```

### Run pytest

```bash
pytest -q
```

## API

### `POST /ask`

Request:

```json
{
  "query": "What is the weather in Karachi?"
}
```

Example response:

```json
{
  "answer": "I used the get_weather tool with local vector retrieval to answer your question.",
  "sources": [],
  "latency_ms": {
    "total": 42,
    "by_step": {
      "retrieve": 12,
      "llm": 0
    }
  },
  "tokens": {
    "prompt": 0,
    "completion": 0
  },
  "tool_trace": [
    {
      "tool": "get_weather",
      "args": {
        "city": "Karachi",
        "query": "What is the weather in Karachi?"
      },
      "latency_ms": 12,
      "cache_hit": false,
      "ok": true
    }
  ]
}
```

### `GET /stream?query=...`

Streams partial answer tokens through Server-Sent Events and ends with the final structured result.

Example:

```bash
curl "http://localhost:5000/stream?query=What%20is%20the%20weather%20in%20Karachi%3F"
```

## Tool behavior

### `web_search`

- Calls DuckDuckGo Instant Answer API
- Extracts abstract and related result links
- Returns only public URLs after filtering

### `get_weather`

- Uses a local FAISS-backed weather store
- Retrieves the closest weather record for the requested city
- Returns a dummy weather result for demonstration purposes

## Notes and limitations

- Weather data is dummy local data, not live weather.
- DuckDuckGo Instant Answer API may return limited results depending on the query.
- The app expects Ollama to be running and reachable at `OLLAMA_URL`.
- The weather tool requires the FAISS index files to exist first.
- The current code does not implement a NumPy fallback for weather retrieval. FAISS and the built weather index are required.

## Docker

Build the image:

```bash
docker build -t mini-agent .
```

Run the container:

```bash
docker run --rm -p 5000:5000 -e OLLAMA_URL=http://host.docker.internal:11434/api/chat -e OLLAMA_MODEL=llama3.1:8b mini-agent
```

On macOS/Linux, use:

```bash
docker run --rm -p 5000:5000 -e OLLAMA_URL=http://host.docker.internal:11434/api/chat -e OLLAMA_MODEL=llama3.1:8b mini-agent
```

Make sure Ollama is running on the host machine before starting the container.
