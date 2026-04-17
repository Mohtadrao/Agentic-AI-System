import json
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import requests
from flask import Flask, Response, jsonify, render_template_string, request
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

try:
    import jsonschema
except Exception:
    jsonschema = None

STORE_DIR = "weather_store"
WEATHER_INDEX_PATH = os.path.join(STORE_DIR, "weather.index")
WEATHER_DOCS_PATH = os.path.join(STORE_DIR, "weather_docs.json")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5000"))

REQUEST_TIMEOUT = (8, 60)
OLLAMA_TIMEOUT = (10, 180)
CACHE_TTL_SECONDS = 300
MAX_TOOL_WORKERS = 4
TOOL_SEMAPHORE = threading.Semaphore(2)
EMBED_DIM = 64

FINAL_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "url": {"type": "string"}
                },
                "required": ["name", "url"]
            }
        },
        "latency_ms": {
            "type": "object",
            "properties": {
                "total": {"type": "integer"},
                "by_step": {
                    "type": "object",
                    "properties": {
                        "retrieve": {"type": "integer"},
                        "llm": {"type": "integer"}
                    },
                    "required": ["retrieve", "llm"]
                }
            },
            "required": ["total", "by_step"]
        },
        "tokens": {
            "type": "object",
            "properties": {
                "prompt": {"type": "integer"},
                "completion": {"type": "integer"}
            },
            "required": ["prompt", "completion"]
        },
        "tool_trace": {
            "type": "array",
            "items": {"type": "object"}
        }
    },
    "required": ["answer", "sources", "latency_ms", "tokens", "tool_trace"]
}

TOOL_INPUT_SCHEMAS = {
    "web_search": {
        "type": "object",
        "properties": {"query": {"type": "string", "minLength": 1}},
        "required": ["query"],
        "additionalProperties": False,
    },
    "get_weather": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "minLength": 1},
            "query": {"type": "string"}
        },
        "required": ["city"],
        "additionalProperties": True,
    },
}


WEATHER_DOCS = [
    {
        "city": "Karachi",
        "temperature_c": 31,
        "condition": "Humid",
        "humidity": 68,
        "text": "Karachi weather is humid with temperature 31C and humidity 68 percent.",
    },
    {
        "city": "Lahore",
        "temperature_c": 34,
        "condition": "Sunny",
        "humidity": 40,
        "text": "Lahore weather is sunny with temperature 34C and humidity 40 percent.",
    },
    {
        "city": "Islamabad",
        "temperature_c": 27,
        "condition": "Cloudy",
        "humidity": 52,
        "text": "Islamabad weather is cloudy with temperature 27C and humidity 52 percent.",
    },
    {
        "city": "London",
        "temperature_c": 16,
        "condition": "Rainy",
        "humidity": 74,
        "text": "London weather is rainy with temperature 16C and humidity 74 percent.",
    },
    {
        "city": "New York",
        "temperature_c": 21,
        "condition": "Partly Cloudy",
        "humidity": 58,
        "text": "New York weather is partly cloudy with temperature 21C and humidity 58 percent.",
    },
]


class TTLCache:
    def __init__(self, ttl_seconds: int = 300):
        self.ttl_seconds = ttl_seconds
        self._store: Dict[str, Tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            item = self._store.get(key)
            if not item:
                return None
            expires_at, value = item
            if expires_at < time.time():
                self._store.pop(key, None)
                return None
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = (time.time() + self.ttl_seconds, value)


CACHE = TTLCache(CACHE_TTL_SECONDS)


def log_event(event: str, **fields: Any) -> None:
    payload = {
        "ts": int(time.time() * 1000),
        "event": event,
        **fields,
    }
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def make_session() -> requests.Session:
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s = requests.Session()
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": "mini-agent/1.0"})
    return s


HTTP = make_session()


def cache_key(prefix: str, payload: Dict[str, Any]) -> str:
    return prefix + ":" + json.dumps(payload, sort_keys=True, ensure_ascii=False)


def is_public_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if parsed.scheme not in {"http", "https"}:
            return False
        if not host:
            return False
        if host in {"localhost"}:
            return False
        if re.match(r"^\d+\.\d+\.\d+\.\d+$", host):
            parts = [int(x) for x in host.split(".")]
            if parts[0] == 10:
                return False
            if parts[0] == 127:
                return False
            if parts[0] == 192 and parts[1] == 168:
                return False
            if parts[0] == 172 and 16 <= parts[1] <= 31:
                return False
        return True
    except Exception:
        return False


def unique_sources(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen_urls = set()
    out = []
    for s in items:
        name = (s or {}).get("name", "").strip() or "Source"
        url = (s or {}).get("url", "").strip()
        if not url or not is_public_url(url):
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)
        out.append({"name": name, "url": url})
    return out


def validate_with_schema(instance: Dict[str, Any], schema: Dict[str, Any]) -> None:
    if jsonschema is not None:
        jsonschema.validate(instance=instance, schema=schema)


def validate_final_output(data: Dict[str, Any]) -> Dict[str, Any]:
    validate_with_schema(data, FINAL_OUTPUT_SCHEMA)
    required_top = {"answer", "sources", "latency_ms", "tokens", "tool_trace"}
    missing = required_top - set(data.keys())
    if missing:
        raise ValueError(f"Missing keys: {sorted(missing)}")
    return data


def parse_tool_arguments(raw_args: Any) -> Dict[str, Any]:
    if raw_args is None:
        return {}
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        raw_args = raw_args.strip()
        if not raw_args:
            return {}
        try:
            parsed = json.loads(raw_args)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    raise ValueError("Tool arguments must be an object or JSON object string")


def hash_token(token: str, dim: int = EMBED_DIM) -> int:
    return sum((i + 1) * ord(c) for i, c in enumerate(token)) % dim


def embed_text(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    if not tokens:
        return vec
    for tok in tokens:
        vec[hash_token(tok, dim)] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.astype(np.float32)


class WeatherVectorStore:
    def __init__(self, docs_path: str, index_path: str):
        self.docs_path = docs_path
        self.index_path = index_path
        self.docs: List[Dict[str, Any]] = []
        self.index = None
        self.backend = "unavailable"

    def load(self) -> None:
        if not os.path.exists(self.docs_path):
            raise FileNotFoundError(
                f"Missing weather docs file: {self.docs_path}. Run build_faiss_db.py first."
            )

        if not os.path.exists(self.index_path):
            raise FileNotFoundError(
                f"Missing weather index file: {self.index_path}. Run build_faiss_db.py first."
            )

        if faiss is None:
            raise RuntimeError("faiss is not installed. Install faiss-cpu to use the weather DB.")

        with open(self.docs_path, "r", encoding="utf-8") as f:
            self.docs = json.load(f)

        self.index = faiss.read_index(self.index_path)
        self.backend = "faiss"

    def search(self, query: str, top_k: int = 1) -> List[Tuple[Dict[str, Any], float]]:
        if self.index is None:
            raise RuntimeError("Weather vector store is not loaded.")

        qv = embed_text(query).reshape(1, -1).astype(np.float32)
        scores, idxs = self.index.search(qv, top_k)

        out = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx >= 0:
                out.append((self.docs[int(idx)], float(score)))
        return out


WEATHER_STORE = WeatherVectorStore(
    docs_path=WEATHER_DOCS_PATH,
    index_path=WEATHER_INDEX_PATH,
)

try:
    WEATHER_STORE.load()
except Exception as e:
    print(json.dumps({
        "event": "weather_store_load_failed",
        "error": str(e)
    }), flush=True)


def web_search(query: str) -> Dict[str, Any]:
    validate_with_schema({"query": query}, TOOL_INPUT_SCHEMAS["web_search"])

    key = cache_key("web_search", {"query": query})
    cached = CACHE.get(key)
    if cached:
        return {**cached, "cache_hit": True}

    with TOOL_SEMAPHORE:
        r = HTTP.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1},
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()

    sources: List[Dict[str, str]] = []
    if data.get("AbstractURL"):
        sources.append({
            "name": data.get("Heading") or "DuckDuckGo Abstract",
            "url": data["AbstractURL"],
        })

    related = data.get("RelatedTopics", [])
    related_results = []

    for item in related:
        if isinstance(item, dict) and item.get("Topics"):
            for sub in item.get("Topics", []):
                if isinstance(sub, dict) and sub.get("FirstURL"):
                    related_results.append({
                        "text": sub.get("Text", ""),
                        "url": sub["FirstURL"],
                    })
        elif isinstance(item, dict) and item.get("FirstURL"):
            related_results.append({
                "text": item.get("Text", ""),
                "url": item["FirstURL"],
            })

    for item in related_results[:4]:
        sources.append({
            "name": (item["text"] or "DuckDuckGo Related Result")[:80],
            "url": item["url"],
        })

    result = {
        "query": query,
        "heading": data.get("Heading", ""),
        "abstract": data.get("AbstractText", ""),
        "abstract_url": data.get("AbstractURL", ""),
        "related_results": related_results[:5],
        "sources": unique_sources(sources),
        "cache_hit": False,
    }
    CACHE.set(key, result)
    return result


def get_weather(city: str, query: Optional[str] = None) -> Dict[str, Any]:
    validate_with_schema({"city": city, "query": query or ""}, TOOL_INPUT_SCHEMAS["get_weather"])

    key = cache_key("get_weather", {"city": city.strip().lower(), "query": (query or "").strip().lower()})
    cached = CACHE.get(key)
    if cached:
        return {**cached, "cache_hit": True}

    if WEATHER_STORE.index is None:
        raise RuntimeError("Weather store is not available. Run build_faiss_db.py first.")

    search_text = f"{city} {query or ''}".strip()
    hits = WEATHER_STORE.search(search_text, top_k=1)
    doc, score = hits[0]

    result = {
        "city": city,
        "matched_city": doc["city"],
        "temperature_c": doc["temperature_c"],
        "condition": doc["condition"],
        "humidity": doc["humidity"],
        "source_note": "Dummy weather tool backed by persisted FAISS retrieval",
        "retrieval_backend": WEATHER_STORE.backend,
        "retrieval_score": round(float(score), 4),
        "sources": [],
        "cache_hit": False,
    }
    CACHE.set(key, result)
    return result


TOOL_FUNCTIONS = {
    "web_search": web_search,
    "get_weather": get_weather,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using DuckDuckGo Instant Answer API for fresh or general information.",
            "parameters": TOOL_INPUT_SCHEMAS["web_search"],
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city from the local dummy weather knowledge base.",
            "parameters": TOOL_INPUT_SCHEMAS["get_weather"],
        },
    },
]


def ollama_chat(messages: List[Dict[str, Any]], tools=None, fmt=None, stream: bool = False):
    payload = {"model": MODEL, "messages": messages, "stream": stream}
    if tools:
        payload["tools"] = tools
    if fmt:
        payload["format"] = fmt
    return HTTP.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT, stream=stream)


def pick_fast_path(user_query: str) -> Optional[List[Dict[str, Any]]]:
    q = user_query.lower()
    if "weather" not in q:
        return None

    city = None
    for known in ["karachi", "lahore", "islamabad", "london", "new york"]:
        if known in q:
            city = known.title() if known != "new york" else "New York"
            break

    if city:
        return [{
            "function": {
                "name": "get_weather",
                "arguments": {
                    "city": city,
                    "query": user_query
                }
            }
        }]
    return None


def execute_tool_call(tc: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    func = tc["function"]["name"]
    raw_args = tc["function"].get("arguments", {})
    start = time.perf_counter()

    try:
        args = parse_tool_arguments(raw_args)
        if func not in TOOL_FUNCTIONS:
            raise ValueError(f"Unknown tool: {func}")

        if func in TOOL_INPUT_SCHEMAS:
            validate_with_schema(args, TOOL_INPUT_SCHEMAS[func])

        result = TOOL_FUNCTIONS[func](**args)
    except Exception as e:
        result = {"error": f"{type(e).__name__}: {e}", "sources": []}
        args = raw_args if isinstance(raw_args, dict) else {"raw": raw_args}

    latency_ms = int((time.perf_counter() - start) * 1000)
    return {"name": func, "args": args, "result": result}, latency_ms


def build_weather_answer(weather: Dict[str, Any]) -> str:
    return (
        f"I used the get_weather tool with local vector retrieval ({weather['retrieval_backend']}) "
        f"to match the city record '{weather['matched_city']}'. "
        f"The weather in {weather['matched_city']} is {weather['condition']}, "
        f"{weather['temperature_c']}°C with humidity {weather['humidity']}%."
    )


def run_agent(user_query: str) -> Dict[str, Any]:
    total_start = time.perf_counter()
    retrieve_ms = 0
    llm_ms = 0
    collected_sources: List[Dict[str, str]] = []
    tool_trace: List[Dict[str, Any]] = []
    token_prompt = 0
    token_completion = 0

    log_event("agent_start", query=user_query)

    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a helpful tool-using assistant. "
                "Use tools when needed. Prefer web_search for fresh or general knowledge. "
                "Use get_weather for weather questions. "
                "Final answer must mention which tools were used and stay grounded in provided tool results."
            ),
        },
        {"role": "user", "content": user_query},
    ]

    tool_calls = pick_fast_path(user_query)

    if tool_calls is not None:
        log_event("planner_fast_path", tool_calls=tool_calls)
        tool_results = []
        with ThreadPoolExecutor(max_workers=min(MAX_TOOL_WORKERS, len(tool_calls))) as ex:
            futures = [ex.submit(execute_tool_call, tc) for tc in tool_calls]
            for fut in as_completed(futures):
                trace, ms = fut.result()
                retrieve_ms += ms
                tool_trace.append({
                    "tool": trace["name"],
                    "args": trace["args"],
                    "latency_ms": ms,
                    "cache_hit": bool(trace["result"].get("cache_hit", False)),
                    "ok": "error" not in trace["result"],
                })
                collected_sources.extend(trace["result"].get("sources", []))
                tool_results.append(trace)
                log_event("tool_complete", tool=trace["name"], latency_ms=ms, ok="error" not in trace["result"])

        if len(tool_results) == 1 and tool_results[0]["name"] == "get_weather" and "error" not in tool_results[0]["result"]:
            weather = tool_results[0]["result"]
            result = {
                "answer": build_weather_answer(weather),
                "sources": [],
                "latency_ms": {
                    "total": int((time.perf_counter() - total_start) * 1000),
                    "by_step": {"retrieve": retrieve_ms, "llm": 0},
                },
                "tokens": {"prompt": 0, "completion": 0},
                "tool_trace": tool_trace,
            }
            validate_final_output(result)
            log_event("agent_done", total_ms=result["latency_ms"]["total"])
            return result

    if tool_calls is None:
        llm_start = time.perf_counter()
        resp = ollama_chat(messages=messages, tools=TOOLS)
        llm_ms += int((time.perf_counter() - llm_start) * 1000)
        resp.raise_for_status()
        first = resp.json()
        assistant_msg = first.get("message", {})
        messages.append(assistant_msg)
        tool_calls = assistant_msg.get("tool_calls", [])
        token_prompt += int(first.get("prompt_eval_count", 0) or 0)
        token_completion += int(first.get("eval_count", 0) or 0)
        log_event("planner_llm", tool_calls=tool_calls, llm_ms=llm_ms)

    if tool_calls:
        with ThreadPoolExecutor(max_workers=min(MAX_TOOL_WORKERS, len(tool_calls))) as ex:
            futures = [ex.submit(execute_tool_call, tc) for tc in tool_calls]
            for fut in as_completed(futures):
                trace, ms = fut.result()
                retrieve_ms += ms
                tool_trace.append({
                    "tool": trace["name"],
                    "args": trace["args"],
                    "latency_ms": ms,
                    "cache_hit": bool(trace["result"].get("cache_hit", False)),
                    "ok": "error" not in trace["result"],
                })
                collected_sources.extend(trace["result"].get("sources", []))
                messages.append({
                    "role": "tool",
                    "name": trace["name"],
                    "content": json.dumps(trace["result"], ensure_ascii=False),
                })
                log_event("tool_complete", tool=trace["name"], latency_ms=ms, ok="error" not in trace["result"])

    final_prompt = """
Return only JSON matching the provided schema.
Requirements:
- answer: concise and explicitly mention which tools were used
- sources: only real public sources actually returned by tools
- if a tool failed, do not invent facts
- latency_ms and tokens can be placeholders; they will be overwritten
- tool_trace can be brief; it will be overwritten
""".strip()
    messages.append({"role": "user", "content": final_prompt})

    llm_start = time.perf_counter()
    final_resp = ollama_chat(messages=messages, fmt=FINAL_OUTPUT_SCHEMA)
    llm_ms += int((time.perf_counter() - llm_start) * 1000)
    final_resp.raise_for_status()
    final_json = final_resp.json()
    token_prompt += int(final_json.get("prompt_eval_count", 0) or 0)
    token_completion += int(final_json.get("eval_count", 0) or 0)

    content = final_json.get("message", {}).get("content", "{}")
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        result = {
            "answer": content.strip() or "No answer returned.",
            "sources": [],
            "latency_ms": {"total": 0, "by_step": {"retrieve": 0, "llm": 0}},
            "tokens": {"prompt": 0, "completion": 0},
            "tool_trace": tool_trace,
        }

    result["sources"] = unique_sources(collected_sources + result.get("sources", []))
    result["tool_trace"] = tool_trace
    result["tokens"] = {"prompt": token_prompt, "completion": token_completion}
    result["latency_ms"] = {
        "total": int((time.perf_counter() - total_start) * 1000),
        "by_step": {"retrieve": retrieve_ms, "llm": llm_ms},
    }

    validate_final_output(result)
    log_event("agent_done", total_ms=result["latency_ms"]["total"])
    return result


def stream_agent(user_query: str):
    try:
        result = run_agent(user_query)
        text = result["answer"]
        for chunk in text.split():
            yield f"data: {json.dumps({'token': chunk + ' '})}\n\n"
        yield f"data: {json.dumps({'done': True, 'result': result})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'done': True, 'error': str(e)})}\n\n"


HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Mini Tool Agent</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root { color-scheme: dark; }
    body {
      margin: 0; font-family: Inter, Arial, sans-serif;
      background: linear-gradient(135deg, #0f172a, #111827 45%, #1e293b);
      color: #e5e7eb; min-height: 100vh;
    }
    .wrap { max-width: 980px; margin: 32px auto; padding: 20px; }
    .card {
      background: rgba(17,24,39,.84); border: 1px solid rgba(255,255,255,.08);
      border-radius: 18px; box-shadow: 0 20px 50px rgba(0,0,0,.3); overflow: hidden;
    }
    .hero { padding: 22px 24px 10px; }
    h1 { margin: 0 0 8px; font-size: 28px; }
    p { color: #94a3b8; line-height: 1.5; }
    .row { display: grid; grid-template-columns: 1fr auto auto; gap: 10px; padding: 18px 24px 24px; }
    textarea {
      width: 100%; min-height: 92px; resize: vertical; border-radius: 14px; border: 1px solid #334155;
      background: #0b1220; color: #e5e7eb; padding: 14px; font-size: 15px;
    }
    button {
      border: 0; border-radius: 12px; padding: 0 16px; cursor: pointer; font-weight: 700;
      background: #2563eb; color: white;
    }
    button.alt { background: #334155; }
    .out { padding: 0 24px 24px; }
    .panel {
      background: #0b1220; border: 1px solid #233047; border-radius: 14px; padding: 14px; margin-top: 14px;
    }
    pre {
      white-space: pre-wrap; word-wrap: break-word; font-size: 13px;
      background: #020617; border-radius: 12px; padding: 14px; border: 1px solid #1e293b;
      overflow-x: auto;
    }
    .chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
    .chip {
      background: #172554; border: 1px solid #1d4ed8; color: #bfdbfe;
      padding: 6px 10px; border-radius: 999px; font-size: 12px;
    }
    a { color: #93c5fd; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="hero">
        <h1>Mini Tool Agent</h1>
        <p>Single-file Flask UI + Ollama agent with retries, cache, structured JSON, source filtering, tool logging, and lightweight vector retrieval for weather.</p>
      </div>
      <div class="row">
        <textarea id="q" placeholder="Ask something... try: What's the weather in Karachi?"></textarea>
        <button onclick="ask()">Ask</button>
        <button class="alt" onclick="streamAsk()">Stream</button>
      </div>
      <div class="out">
        <div class="panel"><strong>Answer</strong><pre id="answer">Ready.</pre></div>
        <div class="panel"><strong>Sources</strong><div id="sources">—</div></div>
        <div class="panel"><strong>Metrics</strong><div class="chips" id="chips"></div></div>
        <div class="panel"><strong>Raw JSON</strong><pre id="raw">{}</pre></div>
      </div>
    </div>
  </div>

<script>
function setResult(data) {
  document.getElementById('answer').textContent = data.answer || '(no answer)';
  document.getElementById('raw').textContent = JSON.stringify(data, null, 2);

  const src = document.getElementById('sources');
  src.innerHTML = '';
  if (!data.sources || !data.sources.length) src.textContent = '—';
  else data.sources.forEach(s => {
    const div = document.createElement('div');
    div.innerHTML = `<a href="${s.url}" target="_blank" rel="noreferrer">${s.name}</a>`;
    src.appendChild(div);
  });

  const chips = document.getElementById('chips');
  chips.innerHTML = '';
  const vals = [
    `Total: ${data.latency_ms?.total ?? 0} ms`,
    `Retrieve: ${data.latency_ms?.by_step?.retrieve ?? 0} ms`,
    `LLM: ${data.latency_ms?.by_step?.llm ?? 0} ms`,
    `Prompt tokens: ${data.tokens?.prompt ?? 0}`,
    `Completion tokens: ${data.tokens?.completion ?? 0}`,
    `Tools: ${(data.tool_trace || []).length}`
  ];
  vals.forEach(v => {
    const c = document.createElement('div');
    c.className = 'chip';
    c.textContent = v;
    chips.appendChild(c);
  });
}

async function ask() {
  const query = document.getElementById('q').value.trim();
  if (!query) return;
  document.getElementById('answer').textContent = 'Thinking...';
  const res = await fetch('/ask', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({query})
  });
  const data = await res.json();
  setResult(data);
}

function streamAsk() {
  const query = document.getElementById('q').value.trim();
  if (!query) return;
  document.getElementById('answer').textContent = '';
  const ev = new EventSource('/stream?query=' + encodeURIComponent(query));
  let acc = '';
  ev.onmessage = (e) => {
    const data = JSON.parse(e.data);
    if (data.token) {
      acc += data.token;
      document.getElementById('answer').textContent = acc;
    }
    if (data.result) setResult(data.result);
    if (data.done) ev.close();
    if (data.error) {
      document.getElementById('answer').textContent = data.error;
      ev.close();
    }
  };
}
</script>
</body>
</html>
"""

app = Flask(__name__)


@app.get("/")
def home():
    return render_template_string(HTML)


@app.post("/ask")
def ask():
    payload = request.get_json(silent=True) or {}
    query = (payload.get("query") or "").strip()
    if not query:
        return jsonify({"error": "query is required"}), 400
    try:
        return jsonify(run_agent(query))
    except Exception as e:
        log_event("agent_error", error=str(e))
        return jsonify({
            "answer": f"Error: {type(e).__name__}: {e}",
            "sources": [],
            "latency_ms": {"total": 0, "by_step": {"retrieve": 0, "llm": 0}},
            "tokens": {"prompt": 0, "completion": 0},
            "tool_trace": []
        }), 500


@app.get("/stream")
def stream():
    query = (request.args.get("query") or "").strip()
    if not query:
        return Response(
            "data: " + json.dumps({"done": True, "error": "query is required"}) + "\n\n",
            mimetype="text/event-stream",
        )
    return Response(stream_agent(query), mimetype="text/event-stream")


def self_test() -> None:
    assert parse_tool_arguments('{"city":"Karachi"}') == {"city": "Karachi"}
    w = get_weather("Karachi", "what is the weather in karachi today")
    assert "temperature_c" in w
    assert w["matched_city"] == "Karachi"
    assert w["retrieval_backend"] in {"faiss", "numpy"}

    s = unique_sources([
        {"name": "x", "url": "https://duckduckgo.com/a"},
        {"name": "x2", "url": "https://duckduckgo.com/a"},
        {"name": "bad", "url": "http://127.0.0.1:8000"}
    ])
    assert len(s) == 1

    validate_final_output({
        "answer": "ok",
        "sources": [],
        "latency_ms": {"total": 1, "by_step": {"retrieve": 1, "llm": 0}},
        "tokens": {"prompt": 0, "completion": 0},
        "tool_trace": []
    })
    print("self-test passed")


if __name__ == "__main__":
    import sys

    if "--self-test" in sys.argv:
        self_test()
    else:
        print(f"Running on http://{HOST}:{PORT} using model={MODEL}")
        app.run(host=HOST, port=PORT, debug=False, threaded=True)