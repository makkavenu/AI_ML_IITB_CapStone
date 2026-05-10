# Multi-Modal AI Agent

A production-ready scaffold for a **GPT-4o–orchestrated multi-modal AI agent** built with **FastAPI**, **LangGraph**, and **Streamlit**.

---

## Architecture

```
User (Streamlit UI)
        │  POST /api/chat  {message, image_base64?, session_id?}
        ▼
┌─────────────────────────────────────────────────────────┐
│                     FastAPI  (app/)                     │
│                                                         │
│  ① Guardrail scan (input)  ──► block 400 if flagged    │
│  ② Build LangGraph state                               │
│  ③ Invoke agent graph                                  │
│  ④ Guardrail scan (output) ──► redact if flagged       │
│  ⑤ Return ChatResponse                                 │
└──────────────────────────┬──────────────────────────────┘
                           │ ainvoke
                           ▼
┌─────────────────────────────────────────────────────────┐
│              LangGraph StateGraph  (agent/)             │
│                                                         │
│   orchestrator ──(tool_call?)──► tool_executor          │
│        │                              │                 │
│        └──(no tool)──► END    synthesizer ──► END       │
└─────────────────────────────────────────────────────────┘
         GPT-4o                  GPT-4o
         + bound tools
```

### Nodes

| Node | Model | Role |
|---|---|---|
| `orchestrator` | GPT-4o | Reads the conversation and emits a structured tool-call (or answers directly). |
| `tool_executor` | — | Dispatches to the selected stub tool and appends a `ToolMessage`. |
| `synthesizer` | GPT-4o | Reads the full conversation + tool output, writes the final user-facing answer. |

### Tools (stubs — replace return values with real endpoint calls)

| Tool | Backing service | Trigger |
|---|---|---|
| `medical_qa` | MedGemma inference endpoint | Medical / clinical questions |
| `legal_qa` | Pinecone RAG + Qwen via AWS Bedrock | Legal questions |
| `vision_llm` | Qwen3-VL-2B inference endpoint | Image description / VQA |
| `object_detection` | YOLOv12-S inference endpoint | "What objects are in this image?" |

---

## File Structure

```
AI_ML_IITB_CapStone/
├── app/                          # FastAPI service
│   ├── main.py                   # App factory, CORS, lifespan
│   ├── routers/
│   │   └── chat.py               # POST /api/chat
│   └── guardrails/
│       └── guardrail_scanner.py  # scan_text_content() — do not modify
├── agent/                        # LangGraph agent
│   ├── graph.py                  # StateGraph definition
│   └── tools/
│       └── tool_definitions.py   # @tool stubs
├── ui/
│   └── streamlit_app.py          # Streamlit chat UI
├── Dockerfile.api
├── Dockerfile.ui
├── docker-compose.yml
├── requirements.txt              # API dependencies
├── requirements-ui.txt           # UI dependencies
├── .env.example                  # Copy to .env and fill in secrets
└── README.md
```

---

## Quick Start

### 1. Copy and fill in secrets

```bash
cp .env.example .env
# Edit .env — add your OPENAI_API_KEY at minimum
```

### 2. Run with Docker Compose

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| Streamlit UI | <http://localhost:8501> |
| FastAPI docs | <http://localhost:8000/docs> |
| Health check | <http://localhost:8000/health> |

### 3. Run locally (without Docker)

```bash
# Terminal 1 — API
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Terminal 2 — UI
pip install -r requirements-ui.txt
API_URL=http://localhost:8000 streamlit run ui/streamlit_app.py
```

---

## API Reference

### `POST /api/chat`

**Request**

```json
{
  "message": "What are the symptoms of appendicitis?",
  "image_base64": "<optional base64 string>",
  "session_id": "<optional uuid>"
}
```

**Response**

```json
{
  "response": "Appendicitis typically presents with …",
  "tool_used": "medical_qa",
  "guardrail_flagged": false
}
```

**Error codes**

| Code | Meaning |
|---|---|
| `400` | Input blocked by safety guardrails |
| `500` | Internal agent or guardrail processing error |

---

## Guardrails

`app/guardrails/guardrail_scanner.py` exposes a single function:

```python
def scan_text_content(text: str) -> ScanResult:
    ...
```

`ScanResult` has three fields: `flagged: bool`, `reason: str`, `category: str`.

The scanner checks for **prompt-injection** and **harmful-content** patterns using compiled regular expressions. It is called twice per request — on the user's input and on the agent's output.

> **Do not modify `guardrail_scanner.py`.**

---

## Replacing Stubs with Real Endpoints

Each tool in `agent/tools/tool_definitions.py` contains a `# TODO` comment with an example `httpx.AsyncClient` call. Fill in the `*_ENDPOINT_URL` env vars in `.env` and replace the `return` statement with the real HTTP call.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ | GPT-4o access |
| `PINECONE_API_KEY` | legal_qa | Pinecone index access |
| `PINECONE_INDEX` | legal_qa | Pinecone index name |
| `AWS_ACCESS_KEY_ID` | legal_qa | AWS Bedrock access |
| `AWS_SECRET_ACCESS_KEY` | legal_qa | AWS Bedrock secret |
| `AWS_DEFAULT_REGION` | legal_qa | AWS region (default `us-east-1`) |
| `MEDGEMMA_ENDPOINT_URL` | medical_qa | MedGemma inference URL |
| `QWEN_VL_ENDPOINT_URL` | vision_llm | Qwen3-VL-2B inference URL |
| `YOLOV12_ENDPOINT_URL` | object_detection | YOLOv12-S inference URL |

---

## Design Decisions

- **All async** — every FastAPI path, LangGraph node, and tool function is `async`.
- **Type-hinted** — all public functions carry full Python type annotations.
- **Google docstrings** — consistent with the project style guide.
- **`logging` not `print`** — structured log lines written to stdout (Docker-friendly).
- **`try/except` everywhere** — failures in nodes and tools are caught and surfaced as clean HTTP errors.
- **Simple structure** — one package per concern; no unnecessary abstractions.
