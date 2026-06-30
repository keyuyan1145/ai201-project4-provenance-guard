# Provenance Guard — planning_v0.md

---

## Section 1: Architecture Narrative

Here is the complete path a single piece of text takes from the moment a creator submits it to the moment a reader sees a transparency label.

**1. Submission arrives at the Rate Limiter.**
The incoming `POST /submit` request is checked against a per-IP counter before any processing begins. If the caller has exceeded 10 requests in the last 60 seconds, the system returns a `429 Too Many Requests` response immediately and nothing further happens. This gate exists to protect the Groq API from abuse and to keep per-request latency predictable.

**2. Request validation.**
If the request passes rate limiting, the Flask route handler reads the JSON body and checks that two required fields are present: `content` (the text to classify) and `creator_id` (an opaque identifier assigned by the calling platform that lets the platform trace the submission back to the right user). If either is missing or empty, a `400 Bad Request` is returned. A UUID is generated and assigned as the `submission_id` — this is returned to the caller as a reference number and is the key used in all subsequent operations (appeals, audit lookups).

**3. Signal 1 — Statistical Heuristics (always runs, acts as cost gate).**
The raw text is passed to the heuristics module, which runs entirely locally with no API call. It computes four sub-scores: sentence length variance, vocabulary diversity (type-token ratio), density of known AI phrase markers, and punctuation regularity. These are averaged into a single `heuristic_score` (0.0–1.0, where 1.0 = AI-like).

If `heuristic_score < 0.25` the text is strongly human-leaning by all surface measures. In this case the LLM signal is **skipped entirely** to avoid unnecessary API cost. The classifier proceeds with heuristic evidence only and confidence is derived from a single-signal formula.

**4. Signal 2 — LLM Semantic Classifier (conditional on gate).**
If the heuristic score is ≥ 0.25, the raw text is sent to the Groq API. The prompt asks a Llama model to return a probability (0.0–1.0) that the text was AI-generated, along with brief reasoning. The call is retried up to a configurable number of times (default: 3) with exponential backoff.

If all retries fail, the system falls back to heuristics-only mode. The response includes a flag `llm_signal_available: false` and a note explaining that the LLM service was unreachable. In this fallback case the label is always forced to **"uncertain"** regardless of the heuristic score — a single signal is not sufficient grounds for a definitive verdict.

**5. Classifier — combines signals into weighted_score + confidence.**
The classifier takes whichever signals are available and produces two output values:

- `weighted_score`: If both signals ran, this is a weighted blend (LLM 65%, heuristics 35%). If only heuristics ran (gate skipped LLM or LLM failed), weighted_score equals the heuristic score directly. Range: 0.0 (human) to 1.0 (AI).
- `final_confidence_score`: Reflects how certain the system is. When both signals ran, confidence is penalized both for scores near 0.5 and for signal disagreement. When only one signal ran (either because the gate short-circuited or LLM failed), confidence uses a single-signal formula and is capped lower to reflect the reduced evidence.

**6. Label Generator — translates scores into platform-ready text.**
The label generator reads `weighted_score` and `final_confidence_score` and maps them to one of three variants: *high-confidence AI*, *high-confidence human*, or *uncertain*. It produces a single `label` string written in plain language for a non-technical reader, including the confidence percentage. This string is what the platform displays verbatim — the API is responsible for generating it, the platform is responsible for rendering it.

**7. Persistence — append to output files.**
Two JSONL files serve as the persistent record. The submission record (submission_id, creator_id, weighted_score, final_confidence_score, label, llm_score, heuristic_score, status = "classified", timestamp) is appended to `data/submissions.jsonl`. A separate audit log entry (event type = "classification", same signals and scores, timestamp) is appended to `data/audit_log.jsonl`. Both writes happen before the response is returned.

**8. Response returned to caller.**
The API returns a JSON object with six fields: `label_id` (the submission UUID reference number), `weighted_score`, `final_confidence_score`, `label` (the display string), `llm_score`, and `heuristic_score`. The platform stores `label_id` so the creator can reference it in a future appeal.

---

## Section 2: Two Detection Signals

### Signal 1 — LLM Semantic Classifier

**What property it measures:** The holistic probability that the text's style, structure, and semantic patterns match the output distribution of large language models. The LLM evaluator considers things like: whether rhetorical moves follow AI-typical patterns (claim → evidence → summary), whether hedging language appears in AI-characteristic ways, whether the vocabulary is smoothly competent but lacks personal idiosyncrasy, and whether the text has the "too complete" quality of AI generation (every idea resolved, no loose threads).

**Why this property differs between human and AI writing:** AI models are optimized for coherence and helpfulness. This creates a detectable signature: predictable topic progression, consistent register throughout, rare usage errors, and an absence of the subtle inconsistencies that human fatigue, passion, or distraction introduce. Human writing tends to have seams — places where the author changed their mind, repeated themselves, trailed off, or escalated emotionally.

**Blind spot:** Highly edited human writing (polished essays, professional journalism, grant proposals) narrows this gap considerably. If a human spent four hours refining a 500-word piece, it may be structurally indistinguishable from AI output to the evaluator. The signal also degrades on very short texts (under ~100 words) where there isn't enough structure to analyze. Additionally, this signal inherits whatever biases the evaluating model has — it may be systematically better at detecting writing from the same model family it was trained on.

---

### Signal 1 — Statistical Heuristics (runs first, acts as cost gate)

**What property it measures:** Four independent surface features of the text, each producing a 0.0–1.0 score (1.0 = AI-like). The four sub-features run **in parallel** (via `ThreadPoolExecutor`) since they are independent computations on the same string, then their scores are averaged equally into `heuristic_score`.

1. **AI Vocabulary Marker Density** — counts frequency of words and phrases statistically overrepresented in LLM output, normalized per 100 words.
   - Words: `delve, certainly, straightforward, robust, seamless, nuanced, comprehensive, leverage, crucial, notably, invaluable, pivotal`
   - Phrases: `it is worth noting, in conclusion, it's important to, in today's, of course`
   - Sentence starters: `Moreover,`, `Furthermore,`, `Additionally,`

2. **Sentence Length Uniformity** — standard deviation of per-sentence word counts, inverted and normalized. Low variance (uniform sentence lengths) → high AI score.

3. **Punctuation Range** — count of *distinct* punctuation types used beyond period/comma (em-dash, ellipsis, parentheses, exclamation mark, question mark, semicolon, colon). Narrow range → AI-like. Wide variety → human-like.

4. **Structural Opener Patterns** — fraction of sentences that begin with a transitional word: `However, Therefore, In addition, As a result, For example, In contrast, Overall, To summarize`. High fraction → AI-like.

**Gate logic:** If `heuristic_score < 0.25`, the text is strongly human-leaning across all surface measures and the LLM signal is skipped entirely. The classifier proceeds on heuristic evidence only.

**Why these properties differ:** These four dimensions measure different failure modes of AI prose: vocabulary choice (markers), rhythm (uniformity), stylistic personality (punctuation range), and structural signposting (openers). AI models are trained on feedback that rewards clarity and coherence — the emergent behavior is prose that is rhythmically regular, over-uses transitional signposting, reaches for a predictable set of "quality" words, and defaults to standard punctuation. Human writers — especially informal or creative ones — do none of these things consistently.

**Blind spot:** Formal human writing (academic papers, legal briefs, structured reports) will trigger multiple sub-features. A professor writing a structured essay will score elevated on markers and opener patterns. These are exactly the ambiguous cases intended to pass through the gate to the LLM signal for a more nuanced read.

---

## Section 3: The False Positive Problem

**Scenario:** A human writer — a graduate student named Maya — submits a 600-word personal essay about her research. She writes methodically. Her sentences are consistently 15–20 words. She uses "Furthermore" and "In conclusion" as she was taught. Her vocabulary is precise but not idiosyncratic. She spent two hours editing it.

**What the signals see:**
- LLM signal: 0.78. The evaluator finds the text well-structured, rhetorically complete, and lacking personal friction. It leans AI.
- Heuristic signal: 0.72. Low sentence length variance, phrase markers ("Furthermore", "In conclusion"), moderate TTR.

**What the classifier computes — four fields, each with a distinct role:**

| Field | Formula | Value | What it measures |
|---|---|---|---|
| `weighted_score` | `0.65 × llm_score + 0.35 × heuristic_score` | **0.76** | The system's combined verdict on AI authorship (0=human, 1=AI). LLM carries more weight because it reads semantics; heuristics are surface-level. |
| `signal_agreement` | `1 − \|llm_score − heuristic_score\|` | **0.94** | How much the two independent signals corroborate each other (1=perfect agreement, 0=completely opposite). High agreement means both measurement methods are telling the same story. |
| `raw_confidence` | `2 × \|weighted_score − 0.5\|` | **0.52** | How far the blended score is from the midpoint. 0.5 is maximum uncertainty (a coin flip); the `2×` rescales the result to the 0–1 range. Answers: *"How decisive is the combined score?"* |
| `final_confidence_score` | `raw_confidence × signal_agreement` | **0.49** | The system's overall certainty. Multiplication enforces that *both* conditions must hold: the score must be decisive AND the two signals must agree. Either factor being weak pulls the whole confidence down. |

**What the label says:** `final_confidence_score` 0.49 falls below the 0.70 threshold required for a high-confidence verdict. Maya's label variant is **Uncertain**: *"The origin of this content is unclear. Our system detected mixed signals and cannot make a confident determination. Confidence is 49%, which means this result should be interpreted with caution."*

This is the right outcome: the system hedges rather than confidently accusing her. `final_confidence_score` does real work here — it prevents a plausible-but-wrong `weighted_score` of 0.76 from becoming a damning label.

**What if both signals score even higher (0.90)?** Then `raw_confidence` = 2 × 0.40 = 0.80, `signal_agreement` = 1.0, `final_confidence_score` = 0.80 → triggers "high-confidence AI." This is the genuine false positive case. Maya sees: *"This content was likely created with AI assistance. Our system analyzed the text using multiple signals and found strong indicators of AI authorship (confidence: 80%)."*

**How the appeals workflow handles it:**
Maya can submit `POST /appeals` with her `submission_id` (returned in the original response as `label_id`) and a brief written explanation ("I wrote this essay myself over two days. My writing style is formal because I'm a grad student."). The system:
1. Validates the `submission_id` exists in submissions.jsonl.
2. Updates `submissions.status` from `"classified"` to `"under_review"`.
3. Writes an audit log entry with `event_type = "appeal"`, capturing her reasoning and a timestamp.
4. Returns a confirmation with `appeal_id` and the message that her case will be reviewed.

No automated re-classification occurs. A human reviewer (or future automated step) would examine the audit log, see both the original classification signals and Maya's reasoning, and make a final determination.

**What this tells us about design:**
- The confidence threshold (0.70) for triggering a strong label is one of the most consequential numbers in the system. Setting it lower catches more edge cases; setting it higher risks more confident wrong verdicts. 0.70 is a deliberate conservative choice.
- The transparency label must always name the confidence percentage explicitly — not hide it — so users know how much weight to give the verdict.
- The appeals path must be frictionless: `submission_id` + free text, nothing more. Creating friction punishes legitimate human writers.
- The label in "uncertain" state should actively invite the creator to appeal, not just state uncertainty passively.

---

## Section 4: API Surface

### `POST /submit`

**Purpose:** Accept a piece of text, run the full detection pipeline, and return the classification result.

**Request body:**
```json
{
  "content": "string (required) — the text to classify",
  "creator_id": "string (required) — opaque platform-assigned identifier for the submitting user"
}
```

**Success response (200):**
```json
{
  "label_id": "uuid-string",
  "weighted_score": 0.76,
  "final_confidence_score": 0.49,
  "label": "The origin of this content is unclear. Our system detected mixed signals and cannot make a confident determination. Confidence is 49%, which means this result should be interpreted with caution.",
  "llm_score": 0.78,
  "heuristic_score": 0.72
}
```

**Error responses:**
- `400` — missing or empty `content` or `creator_id` field
- `429` — rate limit exceeded (includes `Retry-After` header)
- `500` — Groq API unreachable or returned an unexpected response

**Rate limit:** 10 requests per minute per IP.

---

### `POST /appeals`

**Purpose:** Allow a creator to contest a classification. Captures their reasoning, updates status, and logs the appeal.

**Request body:**
```json
{
  "submission_id": "uuid-string (required)",
  "reasoning": "string (required) — the creator's explanation"
}
```

**Success response (200):**
```json
{
  "appeal_id": "uuid-string",
  "submission_id": "uuid-string",
  "status": "under_review",
  "message": "Your appeal has been received and logged. The original classification has been flagged for review."
}
```

**Error responses:**
- `400` — missing `submission_id` or `reasoning`
- `404` — `submission_id` does not exist in submissions.jsonl

---

### `GET /log`

**Purpose:** Return structured audit log entries so the audit trail is inspectable.

**Query parameters (all optional):**
- `limit` — max entries to return (default: 20)
- `event_type` — filter to `"classification"` or `"appeal"`

**Success response (200):**
```json
{
  "entries": [
    {
      "event_id": "uuid-string",
      "event_type": "classification",
      "submission_id": "uuid-string",
      "creator_id": "string",
      "llm_score": 0.78,
      "heuristic_score": 0.72,
      "llm_signal_available": true,
      "weighted_score": 0.76,
      "signal_agreement": 0.94,
      "raw_confidence": 0.52,
      "final_confidence_score": 0.49,
      "label_variant": "uncertain",
      "label": "The origin of this content is unclear...",
      "created_at": "2026-06-30T14:23:01Z"
    }
  ],
  "total": 1
}
```

---

## Section 5: Architecture Diagram

### Flow 1 — Submission

```
Client
  │
  │  POST /submit  {"content": "...", "creator_id": "..."}
  ▼
┌─────────────────┐
│  Rate Limiter   │──── over limit ──────────────────────► 429 Response
└────────┬────────┘
         │ request allowed
         ▼
┌──────────────────────┐
│  Request Validator   │──── missing content/creator_id ──► 400 Response
└──────────┬───────────┘
           │ content: str, creator_id: str, submission_id: UUID
           ▼
┌──────────────────────────────────────────────┐
│  Signal 1: Heuristic (4 sub-features)        │
│  [vocabulary markers, sentence uniformity,   │
│   punctuation range, opener patterns]        │
│  sub-features run in parallel internally     │
└──────────────────┬───────────────────────────┘
                   │ heuristic_score: float
                   ▼
           ┌───────────────┐
           │  Gate Check   │──── score < 0.25 ──► skip LLM
           └───────┬───────┘                          │
                   │ score >= 0.25                     │
                   ▼                                   │
  ┌────────────────────────────────┐                  │
  │  Signal 2: LLM Classifier      │                  │
  │  (Groq API, with retry logic)  │                  │
  └──────────────┬─────────────────┘                  │
                 │                                     │
        ┌────────┴────────┐                           │
        │ success         │ all retries failed        │
        │ llm_score:float │ llm_signal_available:false│
        └────────┬────────┘                           │
                 └──────────────┬─────────────────────┘
                                │ heuristic_score [+ llm_score if available]
                                ▼
                      ┌──────────────────┐
                      │    Classifier    │
                      │  (weighted blend │
                      │  or single-signal│
                      │  if LLM absent)  │
                      └────────┬─────────┘
                               │ weighted_score: float, final_confidence_score: float
                               ▼
                      ┌──────────────────┐
                      │  Label Generator │
                      └────────┬─────────┘
                               │ label: str  (display-ready text)
                               ▼
                      ┌──────────────────────────────────────┐
                      │  File Write                          │
                      │  data/submissions.jsonl  (append)    │
                      │  data/audit_log.jsonl    (append)    │
                      └────────┬─────────────────────────────┘
                               │ persisted
                               ▼
                      ┌─────────────────────────────────────────────────────┐
                      │  JSON Response                                       │
                      │  {label_id, weighted_score, final_confidence_score,  │
                      │   label, llm_score, heuristic_score}                 │
                      └─────────────────────────────────────────────────────┘
```

---

### Flow 2 — Appeal

```
Client
  │
  │  POST /appeals  {"submission_id": "...", "reasoning": "..."}
  ▼
┌──────────────────────┐
│  Request Validator   │──── missing fields ──► 400 Response
└──────────┬───────────┘
           │ submission_id: str, reasoning: str
           ▼
┌──────────────────────────┐
│  File Lookup             │──── not found ──► 404 Response
│  (submissions.jsonl scan)│
└──────────┬───────────────┘
           │ submission record confirmed
           ▼
┌──────────────────────────┐
│  Status Updater          │  rewrites record: status = "under_review"
│  (submissions.jsonl)     │
└──────────┬───────────────┘
           │ appeal_id: UUID
           ▼
┌──────────────────────────┐
│  Audit Logger            │  appends to audit_log.jsonl
│  (audit_log.jsonl append)│  event=appeal, submission_id, reasoning, appeal_id
└──────────┬───────────────┘
           │ logged
           ▼
┌──────────────────────────────────────────────────────────┐
│  JSON Response                                           │
│  {appeal_id, submission_id, status: "under_review",      │
│   message: "Appeal received and logged..."}              │
└──────────────────────────────────────────────────────────┘
```
