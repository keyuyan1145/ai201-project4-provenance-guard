# Provenance Guard

A Flask-based backend that classifies submitted text as AI-generated or human-written using a two-signal detection pipeline. Returns a confidence-scored transparency label and maintains a full audit log with creator appeal support.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # add your GROQ_API_KEY
flask run
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/submit` | Classify text. Rate-limited. |
| `POST` | `/appeal` | Contest a classification result. |
| `GET` | `/log` | Return recent audit log entries. |

### POST /submit

**Request body:**
```json
{
  "text": "The text to classify.",
  "creator_id": "user-identifier"
}
```

**Response:**
```json
{
  "label_id": "uuid",
  "content_id": "uuid",
  "label": "uncertain",
  "label_text": "The origin of this content is unclear...",
  "weighted_score": 0.42,
  "final_confidence_score": 0.18,
  "attribution": 0.18,
  "heuristic_score": 0.42,
  "llm_score": 0.39,
  "agreement_score": 0.97
}
```

**Label variants:**

| Variant | Condition |
|---|---|
| `high_confidence_ai` | `weighted_score ≥ 0.65` and `confidence ≥ 0.70` |
| `high_confidence_human` | `weighted_score ≤ 0.35` and `confidence ≥ 0.70`, or heuristic gate closed |
| `uncertain` | Everything else |

### POST /appeal

**Request body:**
```json
{
  "content_id": "uuid-from-submit-response",
  "creator_reasoning": "I wrote this myself..."
}
```

**Response:**
```json
{
  "appeal_id": "uuid",
  "content_id": "uuid",
  "status": "under_review",
  "message": "Your appeal has been received and is under review.",
  "timestamp": "2025-04-01T14:32:10.123Z"
}
```

### GET /log

Query params: `?limit=20` (default 20), `?event_type=classification|appeal`

---

## Rate Limiting

Rate limiting is applied to `POST /submit` only.

### Chosen limits

```
10 requests per minute
100 requests per day
```

### Reasoning

**10 per minute** — A writer manually submitting their own work produces one request every few seconds at most: paste text, hit submit, read the result, move on. Ten per minute (one every six seconds) is generous for legitimate human use. A script flooding the endpoint would exhaust this in under a minute; a real user would never notice it.

**100 per day** — A heavy user (a student checking multiple drafts, an editor reviewing several pieces) is unlikely to exceed 100 submissions in a single day. This cap stops automated scraping campaigns that might run overnight but is completely transparent to any individual creator.

**Per-IP enforcement** — Limits are keyed to the client's remote address via `get_remote_address`. This is appropriate for a single-tenant deployment; a production system behind a load balancer would use a forwarded-header key function instead.

**Implementation** (`config.py` → `app.py`):
```python
# config.py
RATE_LIMIT = "10 per minute;100 per day"

# app.py
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)

@app.route("/submit", methods=["POST"])
@limiter.limit(config.RATE_LIMIT)
def submit():
    ...
```

### Evidence — 429 responses at request 11 and 12

The following output was captured by sending 12 rapid `POST /submit` requests in a tight loop. Requests 1–10 return `200`; requests 11–12 are rejected with `429 Too Many Requests`:

```
200
200
200
200
200
200
200
200
200
200
429
429
```

Test command used:
```bash
for i in $(seq 1 12); do
  curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:5000/submit \
    -H "Content-Type: application/json" \
    -d '{"text": "This is a test submission for rate limit testing purposes only.", "creator_id": "ratelimit-test"}'
done
```

---

## Detection Pipeline

### Signal 1 — Statistical Heuristics (`pipeline/heuristic_signal.py`)

Four sub-features run in parallel via `ThreadPoolExecutor`:

| Sub-feature | What it measures |
|---|---|
| `vocab_marker_density` | Frequency of known AI vocabulary markers and transitional phrases per 100 words |
| `sentence_length_uniformity` | Low coefficient of variation in sentence lengths → AI-like uniformity |
| `punctuation_range` | Narrow punctuation variety (AI text avoids em-dashes, semicolons, parentheses) |
| `structural_opener_patterns` | Fraction of sentences beginning with signposting words (Furthermore, Moreover, etc.) |

Scores are averaged into a single `heuristic_score` (0–1).

**Cost gate:** if `heuristic_score < 0.15`, the text is strongly human-leaning and the LLM call is skipped entirely. The label is set to `high_confidence_human` directly.

### Signal 2 — LLM Semantic Classifier (`pipeline/llm_signal.py`)

Calls the Groq API (Llama 3.3 70B) with a structured forensic-linguist prompt. The model returns `{"ai_probability": float, ...}`. Retries up to 3 times with exponential backoff (1s, 2s). Returns `None` if all retries fail.

### Classifier (`pipeline/classifier.py`)

| Mode | Formula |
|---|---|
| Dual-signal | `weighted_score = 0.65 × llm + 0.35 × heuristic`; `agreement = 1 − \|llm − heuristic\|`; `confidence = raw_confidence × agreement` |
| Single-signal | `weighted_score = heuristic_score`; `confidence = raw_confidence × 0.75` |

---

## Running Tests

```bash
python -m pytest tests/ -q
```

195 tests across signal logic, classifier math, endpoint validation, audit log, and appeal flow.
