# SymPrompt Production Deployment Guide

This document describes how to deploy evolved SymPrompt components to a production inference system serving thousands of requests.

## Deployment Options

### Option 1: Static File Replacement (Simple)

Copy evolved programs over the originals:

```bash
cp evolution/openevolve_output_profiles/best/best_program.py symprompt/symil/profiles.py
cp evolution/openevolve_output_router/best/best_program.py symprompt/router/smart_router.py
cp evolution/openevolve_output_translation/best/best_program.py symprompt/translation/pipeline.py
```

Then rebuild and deploy as normal Python package.

### Option 2: Dynamic Loader with OpenAI-Compatible Proxy (Flexible)

More flexible approach that loads evolved programs at runtime without modifying source files.

## Architecture Overview (Option 2)

```
┌─────────────────────────────────────────────────────────────────┐
│                     Load Balancer (nginx/HAProxy)               │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Worker Pod 1   │  │  Worker Pod 2   │  │  Worker Pod N   │
│  ┌───────────┐  │  │  ┌───────────┐  │  │  ┌───────────┐  │
│  │  FastAPI  │  │  │  │  FastAPI  │  │  │  │  FastAPI  │  │
│  │  uvicorn  │  │  │  │  uvicorn  │  │  │  │  uvicorn  │  │
│  └─────┬─────┘  │  │  └─────┬─────┘  │  │  └─────┬─────┘  │
│        │        │  │        │        │  │        │        │
│  ┌─────▼─────┐  │  │  ┌─────▼─────┐  │  │  ┌─────▼─────┐  │
│  │ SymPrompt │  │  │  │ SymPrompt │  │  │  │ SymPrompt │  │
│  │  Pipeline │  │  │  │  Pipeline │  │  │  │  Pipeline │  │
│  │ (evolved) │  │  │  │ (evolved) │  │  │  │ (evolved) │  │
│  └───────────┘  │  │  └───────────┘  │  │  └───────────┘  │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             ▼
                    ┌─────────────────┐
                    │   LLM Backend   │
                    │ (OpenRouter/    │
                    │  self-hosted)   │
                    └─────────────────┘
```

## Key Components

### 1. Stateless Worker Pods

- Each pod loads evolved programs once at startup
- No shared state between pods - horizontal scaling is trivial
- Workers are CPU-bound (solvers) + I/O-bound (LLM calls)

### 2. Module Loading at Startup

```python
import sys
import importlib.util

def load_evolved_module(path: str, module_name: str):
    """Load a module from file path and register in sys.modules."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Load evolved programs once when worker starts
best_profiles = load_evolved_module(
    "evolution/openevolve_output_profiles/best/best_program.py",
    "candidate_profiles"
)
best_router = load_evolved_module(
    "evolution/openevolve_output_router/best/best_program.py",
    "candidate_router"
)
best_translation = load_evolved_module(
    "evolution/openevolve_output_translation/best/best_program.py",
    "candidate_translation"
)

# Override sys.modules so imports use evolved versions
sys.modules["symprompt.symil.profiles"] = best_profiles
sys.modules["symprompt.router.smart_router"] = best_router
sys.modules["symprompt.translation.pipeline"] = best_translation
```

### 3. Request Flow

```
Client → /v1/chat/completions → Route → Translate → Solve → Response
                                  │         │          │
                            SmartRouter  Pipeline   Z3/Clingo/
                            (evolved)   (evolved)   Scallop
```

### 4. Scaling Strategy

| Component | Scaling Approach |
|-----------|------------------|
| CPU-bound work (Z3/Clingo solvers) | 1 worker per CPU core |
| LLM calls | Async I/O, many concurrent requests per worker |
| Target throughput | ~50-200 concurrent requests per pod |

### 5. Deployment Updates

When you run evolution and get better programs:

1. Build new Docker image with updated `best_program.py` files
2. Rolling deployment - new pods pick up evolved programs
3. Zero-downtime update via Kubernetes rolling strategy

## FastAPI Proxy Implementation

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import sys
import importlib.util

app = FastAPI()

# Pydantic models for OpenAI-compatible API
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    choices: List[dict]
    usage: dict

def load_evolved_module(path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

@app.on_event("startup")
def load_evolved_programs():
    """Load evolved modules at startup (once per worker)."""
    # Load and override modules with evolved versions
    profiles = load_evolved_module(
        "evolution/openevolve_output_profiles/best/best_program.py",
        "evolved_profiles"
    )
    sys.modules["symprompt.symil.profiles"] = profiles

    router = load_evolved_module(
        "evolution/openevolve_output_router/best/best_program.py",
        "evolved_router"
    )
    sys.modules["symprompt.router.smart_router"] = router

    translation = load_evolved_module(
        "evolution/openevolve_output_translation/best/best_program.py",
        "evolved_translation"
    )
    sys.modules["symprompt.translation.pipeline"] = translation

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest) -> ChatResponse:
    """OpenAI-compatible chat completions endpoint."""
    # Import after module override
    from symprompt.integration.router_adapter import route_with_escalation

    # Get the last user message
    user_message = next(
        (m.content for m in reversed(request.messages) if m.role == "user"),
        ""
    )

    # Route through SymPrompt pipeline
    result = await route_with_escalation(user_message)

    return ChatResponse(
        id="chatcmpl-" + str(hash(user_message))[:8],
        choices=[{
            "index": 0,
            "message": {"role": "assistant", "content": result},
            "finish_reason": "stop"
        }],
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)
```

## Capacity Planning

For 1000 req/s with average 100ms latency:

| Metric | Value |
|--------|-------|
| Concurrent workers needed | ~100 |
| Workers per pod | 4 |
| Total pods | ~25 |
| CPU usage | Low (most time waiting on LLM API) |

The bottleneck is typically the upstream LLM provider, not SymPrompt's symbolic reasoning.

## Dockerfile Example

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for solvers
RUN apt-get update && apt-get install -y \
    libz3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY symprompt/ symprompt/
COPY evolution/ evolution/

# Copy the proxy server
COPY proxy_server.py .

EXPOSE 8000

CMD ["uvicorn", "proxy_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

## Kubernetes Deployment Example

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: symprompt-proxy
spec:
  replicas: 25
  selector:
    matchLabels:
      app: symprompt-proxy
  template:
    metadata:
      labels:
        app: symprompt-proxy
    spec:
      containers:
      - name: proxy
        image: symprompt-proxy:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            cpu: "1"
            memory: "2Gi"
          limits:
            cpu: "2"
            memory: "4Gi"
        env:
        - name: SYMPROMPT_LLM_MODEL
          value: "openrouter/x-ai/grok-4.1-fast:free"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 25%
      maxUnavailable: 25%
---
apiVersion: v1
kind: Service
metadata:
  name: symprompt-proxy
spec:
  selector:
    app: symprompt-proxy
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Monitoring Recommendations

1. **Metrics to track:**
   - Request latency (p50, p95, p99)
   - Tier distribution (% Tier 0 vs Tier 1 vs Tier 2)
   - Solver success rates
   - LLM API latency and errors

2. **Alerting thresholds:**
   - p95 latency > 500ms
   - Error rate > 1%
   - LLM API error rate > 5%

3. **Logging:**
   - Log routing decisions for debugging
   - Log solver failures with full context
   - Structured JSON logs for aggregation
