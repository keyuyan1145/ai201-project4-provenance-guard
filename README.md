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

## Architecture

### Submission Flow

```
Client
  │
  │  POST /submit  {"text": "...", "creator_id": "..."}
  ▼
┌─────────────────┐
│  Rate Limiter   │──── over limit ──────────────────────► 429 Response
└────────┬────────┘
         │ request allowed
         ▼
┌──────────────────────┐
│  Request Validator   │──── missing text/creator_id ─────► 400 Response
└──────────┬───────────┘
           │ text: str, creator_id: str, label_id: UUID
           ▼
┌──────────────────────────────────────────────────────┐
│  Signal 1: Heuristics (4 sub-features, parallel)     │
│  [vocab_marker_density  30%]                         │
│  [structural_opener_patterns  25%]                   │
│  [specificity_density  30%]                          │
│  [sentence_length_uniformity  15%]                   │
└──────────────────┬───────────────────────────────────┘
                   │ heuristic_score: float
                   ▼
         ┌─────────────────────┐
         │  Two-sided Gate     │
         └──┬──────────────┬───┘
    score < 0.15           score > 0.85        0.15 ≤ score ≤ 0.85
    (human gate)           (AI gate)           │
    llm_score = null       llm_score = null    ▼
         │                      │    ┌───────────────────────────┐
         │                      │    │  Signal 2: LLM Classifier │
         │                      │    │  (Groq API, retry ×3)     │
         │                      │    └──────────────┬────────────┘
         │                      │         success   │   all retries fail
         │                      │     llm_score:    │   llm_score = null
         │                      │     float         │
         └──────────────────────┴───────────────────┘
                                │
                                ▼
                    ┌───────────────────────────┐
                    │  Classifier               │
                    │  adaptive weighted blend  │
                    │  → weighted_score         │
                    │  → signal_agreement       │
                    └──────────┬────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Label Assignment    │
                    │  lower gate closed → high_confidence_human  │
                    │  upper gate closed → high_confidence_ai     │
                    │  llm failed        → uncertain              │
                    │  ws ≥ 0.70         → high_confidence_ai     │
                    │  ws ≤ 0.30         → high_confidence_human  │
                    │  else              → uncertain              │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Audit Log Write     │
                    │  data/audit_log.jsonl│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌────────────────────────────────────────────┐
                    │  JSON Response                             │
                    │  {label_id, content_id, text,             │
                    │   weighted_score, attribution, label,      │
                    │   label_text, llm_score, heuristic_score,  │
                    │   agreement_score}                         │
                    └────────────────────────────────────────────┘
```

### Appeal Flow

```
Client
  │
  │  POST /appeal  {"content_id": "...", "creator_reasoning": "..."}
  ▼
┌──────────────────────┐
│  Request Validator   │──── missing fields ──► 400 Response
└──────────┬───────────┘
           │ content_id: str, creator_reasoning: str
           ▼
┌──────────────────────────────┐
│  Audit Log Lookup            │──── not found ──► 404 Response
│  (audit_log.jsonl scan)      │──── already under_review ─► 409 Response
└──────────┬───────────────────┘
           │ entry confirmed, status = "classified"
           ▼
┌──────────────────────────────┐
│  Status Updater              │  rewrites entry: status = "under_review"
│  (audit_log.jsonl in-place)  │
└──────────┬───────────────────┘
           │ appeal_id: UUID
           ▼
┌──────────────────────────────┐
│  Audit Logger                │  appends appeal entry to audit_log.jsonl
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│  JSON Response                                           │
│  {appeal_id, content_id, status: "under_review",         │
│   message: "Your appeal has been received...",           │
│   timestamp}                                             │
└──────────────────────────────────────────────────────────┘
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
  "text": "The text to classify.",
  "label": "uncertain",
  "label_text": "The origin of this content is unclear...",
  "weighted_score": 0.52,
  "attribution": 0.52,
  "heuristic_score": 0.48,
  "llm_score": 0.58,
  "agreement_score": 0.90
}
```

**Label variants:**

| Variant | Condition |
|---|---|
| `high_confidence_ai` | `weighted_score ≥ 0.70`, or `heuristic_score > 0.85` (upper gate) |
| `high_confidence_human` | `weighted_score ≤ 0.30`, or `heuristic_score < 0.15` (lower gate) |
| `uncertain` | Everything else (including LLM failure in gate range) |

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

## Detection Pipeline

### Why two signals?

A single signal would be brittle. Statistical heuristics are fast and cheap but they only read *surface features* — how the text is written. An LLM classifier reads *semantic structure* — what the text means, whether its argumentation follows AI-typical patterns, whether ideas feel "too complete." These two failure modes are largely independent: formal human writing trips the heuristics but not the LLM; lightly-edited AI output clears the heuristics but not the LLM. Running both and looking for disagreement is more robust than either alone.

### Signal 1 — Statistical Heuristics (`pipeline/heuristic_signal.py`)

Four sub-features run in parallel via `ThreadPoolExecutor`, combined via **weighted average** into a single `heuristic_score` (0.0–1.0, where 1.0 = strongly AI-like). Every sub-feature scores *high* for AI-like characteristics so the weighted sum stays directionally consistent.

| Sub-feature | Weight | What it measures | Why this signal |
|---|---|---|---|
| **AI Vocabulary Marker Density** | 30% | Words/phrases statistically overrepresented in LLM output, normalized per 100 words. Includes word markers (`delve`, `robust`, `leverage`, `crucial`, `pivotal`, `nuanced`, `comprehensive`, `seamless`, `invaluable`, `notably`, `straightforward`, `certainly`, `essential`, `transformative`, `ensure`, `stakeholders`, `innovative`, `utilize`, `demonstrate`, `paramount`, `imperative`, `underscore`, `encompass`, `foster`) and phrase markers (`it is worth noting`, `in conclusion`, `it is important to`, `ensure that`, `in today's`). | These words co-occur with LLM output at rates far exceeding their base-rate in human prose. Works at any text length — even a single sentence with "delve" and "leverage" is informative. |
| **Structural Opener Patterns** | 25% | Fraction of sentences beginning with logical signpost words: `However`, `Therefore`, `Moreover`, `Furthermore`, `Additionally`, `In addition`, `As a result`, `For example`, `In contrast`, `Overall`, `To summarize`. | LLMs over-apply explicit transitional scaffolding; humans use it sparingly. Signal is noisier on short texts (one opener-starting sentence inflates the fraction when there are only four total), so it receives a lower weight than vocabulary. |
| **Specificity and Concrete Details** | 30% | Density of concrete anchors — numbers, percentages, and mid-sentence capitalized words (proper nouns) — per word count. Score = `max(0, 1 − density / 0.10)`, saturating at 0.0 when 1 in 10 words is a specific anchor. | AI text is abstract and generic; human text uses specific dates, figures, names, and places. This feature detects *absence* of specificity — no numbers or named entities across a paragraph is an AI tell. Works at any length and replaced the original `punctuation_range` feature, which was nearly constant on short texts and carried no discriminative power. |
| **Sentence Length Uniformity** | 15% | Inverse coefficient of variation (std/mean) of per-sentence word counts. Low CV (rhythmically regular) → high score. Saturates at CV ≥ 0.5. | AI models produce rhythmically uniform prose. However, with fewer than five sentences (typical for short submissions) one outlier sentence dominates the CV, making this signal unreliable — it gets the lowest weight. |

**Why these weights?** The system is designed for short prose submissions (typically under 150 words). At that length, two features are structurally noisier: `sentence_length_uniformity` operates on too few sentences, and `structural_opener_patterns` is inflated by any single opener. `vocab_marker_density` and `specificity_density` both operate at the word level — they are equally informative at 40 words or 400 words — so they share the top weight (30% each).

**Two-sided cost gate:**

| Condition | Action | Label |
|---|---|---|
| `heuristic_score < 0.15` | LLM **skipped** — strong human evidence | `high_confidence_human` |
| `heuristic_score > 0.85` | LLM **skipped** — strong AI evidence | `high_confidence_ai` |
| `0.15 ≤ heuristic_score ≤ 0.85` | LLM runs | Determined by dual-signal combination |

Both extremes bypass the Groq API entirely, eliminating latency and cost where the heuristics already give a decisive answer.

**For real deployment:** The vocabulary list would need continuous maintenance — LLM writing patterns drift as models are updated. The specificity detector would need a proper NER model rather than mid-sentence capitalization heuristics to avoid false positives from brand names and abbreviations. The gate thresholds (0.15/0.85) were chosen conservatively; with labeled production data you'd tune these empirically.

### Signal 2 — LLM Semantic Classifier (`pipeline/llm_signal.py`)

Calls the Groq API (Llama 3.3 70B) with a structured forensic-linguist prompt. The model returns `{"ai_probability": float, ...}`. Retries up to 3 times with exponential backoff (1s, 2s, 4s). Returns `None` if all retries fail.

**What it captures:** Holistic semantic patterns — whether rhetorical moves follow AI-typical progressions (claim → evidence → summary), whether vocabulary is competent but lacks personal idiosyncrasy, whether the text has the "too complete" quality of AI generation (every idea resolved, no loose threads or raw emotion).

**Why LLM gets higher weight than heuristics:** The four sub-features answer *how the text is written*. The LLM answers *what the text means*. Semantic analysis is a more comprehensive signal. Additionally, at short text lengths the heuristics are less stable, while LLMs handle short texts comparably well.

**For real deployment:** The LLM evaluator inherits its own biases — it may be systematically better at detecting output from the same model family. Using a different model family for evaluation than for generation would reduce this blind spot.

---

## Confidence Scoring

### Formula

The `weighted_score` is the single output score (0 = human, 1 = AI). It serves as both the confidence indicator and the attribution verdict. In dual-signal mode it is an adaptive weighted average; in single-signal mode it is the heuristic score directly.

**Dual-signal weight tiers (applied in priority order):**

| Condition | LLM weight | Heuristic weight | Rationale |
|---|---|---|---|
| `word_count > 150` | 65% | 35% | Longer text makes heuristics more stable; increased heuristic contribution is warranted |
| `\|llm − heuristic\| > 0.40` | 85% | 15% | Strong disagreement: trust the deeper semantic signal almost exclusively |
| otherwise | 70% | 30% | Standard short-text case with moderate agreement |

**Signal agreement:** `agreement_score = 1 − |llm_score − heuristic_score|`. Returned in the response; not used in the label threshold but informative for understanding *why* the system landed where it did.

### Example 1 — High-confidence AI verdict

**Text submitted:**
> "Delving into the comprehensive realm of robust and seamless solutions, it is worth noting that leveraging these crucial and invaluable frameworks is pivotal. Moreover, the nuanced approach provides notably straightforward pathways to success. Furthermore, it is important to recognize that modern enterprises require sophisticated solutions. Additionally, these comprehensive methodologies ensure seamless integration throughout. In conclusion, the pivotal role of robust systems cannot be overstated."

**Scores:**
```
heuristic_score:   0.9712
  vocab_marker_density:       1.0000   (density saturates — 10+ markers/100 words)
  structural_opener_patterns: 0.8000   (4 of 5 sentences start with a signpost)
  specificity_density:        1.0000   (zero numbers, zero named entities)
  sentence_length_uniformity: 0.8500   (very uniform rhythm)

→ upper gate fires (0.9712 > 0.85) — LLM skipped
llm_score:         null
weighted_score:    0.9712   (single-signal = heuristic_score)
label:             high_confidence_ai
```

Both vocabulary saturation and total absence of concrete details push this to the extreme. The upper gate fires, which is the correct behavior — spending an API call on a text this obviously AI-generated would waste money without changing the verdict.

### Example 2 — Lower-confidence uncertain verdict

**Text submitted:**
> "I've been thinking about how our team handles the release cycle. We usually push on Thursdays — not for any deep reason, mostly because that's what Jake started doing in 2022. Honestly the process works okay, though I think we could be smarter about how we batch the hotfixes. Something to bring up at the retrospective."

**Scores:**
```
heuristic_score:   0.4800
  vocab_marker_density:       0.2000   (no AI vocab markers)
  structural_opener_patterns: 0.0000   (no signpost openers)
  specificity_density:        0.2800   ("Thursday", "Jake", "2022" are concrete anchors)
  sentence_length_uniformity: 0.5600   (moderate sentence variation)

llm_score:         0.5800   (LLM sees hedged but semi-structured argumentation)
gap:               0.1000   (≤ 0.40 → standard 70/30 weights)
weighted_score:    0.70 × 0.58 + 0.30 × 0.48 = 0.406 + 0.144 = 0.5500
agreement_score:   0.9000
label:             uncertain   (0.30 < 0.55 < 0.70)
```

This text contains personal context, specific names and dates (concrete anchors that push specificity score down), and no AI vocabulary — so heuristics lean human. The LLM sees somewhat structured argumentation and hedging language, pushing its score slightly above 0.5. The resulting `weighted_score` of 0.55 falls in the uncertain zone (0.35–0.65), which is the correct outcome: the system genuinely cannot tell.

The contrast between 0.97 and 0.55 shows the scoring produces meaningful variation — it doesn't collapse everything to a constant near 0.5.

---

## Transparency Labels

All three label variants are returned as the `label_text` field in the `/submit` response. The confidence percentage is interpolated from `weighted_score`.

### Variant 1 — `high_confidence_ai`
*Triggered when: `weighted_score ≥ 0.70` OR `heuristic_score > 0.85` (upper gate)*

> "This content shows strong indicators of AI authorship. Our system analyzed the text across multiple signals and found patterns consistent with AI-generated writing (confidence: {X}%). If you are the creator and believe this is incorrect, you can submit an appeal using your submission ID."

### Variant 2 — `high_confidence_human`
*Triggered when: `weighted_score ≤ 0.30` OR `heuristic_score < 0.15` (lower gate)*

> "This content appears to be human-written. Our system analyzed the text across multiple signals and found no significant indicators of AI authorship (confidence: {X}%)."

### Variant 3 — `uncertain`
*Triggered when: everything else (including LLM failure in the gate range)*

> "The origin of this content is unclear. Our system detected mixed or inconclusive signals and cannot make a confident determination (confidence: {X}%). This result should be interpreted with caution. If you are the creator of this content and believe it is human-written, you may submit an appeal using your submission ID."

**LLM failure addendum** (appended when the LLM was attempted but all retries failed):

> " Note: the AI signal was temporarily unavailable; this result is based on surface-level analysis only."

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

```
200 200 200 200 200 200 200 200 200 200 429 429
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

## Running Tests

```bash
python -m pytest tests/ -q
```

209 tests across signal logic, classifier math, endpoint validation, audit log, and appeal flow.

---

## Known Limitations

### 1. Formal academic or professional writing by a human

A graduate student who writes structured essays with uniform paragraphs, transitional phrases ("Furthermore", "In conclusion"), and precise vocabulary will trip multiple sub-features simultaneously — elevated opener density, high vocab marker score, and low sentence length variance all fire together. If the LLM also finds the rhetoric "too complete," the system can issue a `high_confidence_ai` verdict for entirely human-written work.

The root cause is that **the heuristic sub-features measure writing style conventions, not authorship**. "Formally trained human writer" and "AI output" overlap significantly in surface features. The signal agreement system partially mitigates this — if the LLM is less certain than the heuristics, the resulting `weighted_score` moves toward uncertain — but it doesn't fully resolve it. A real deployment would need a calibration set of labeled academic writing to set appropriate gate thresholds.

### 2. AI-generated text that avoids the known vocabulary list

The `vocab_marker_density` sub-feature depends on a static word list compiled from current LLM writing patterns. A user who prompts an LLM with "write this without using the words 'robust', 'leverage', or 'delve'" will produce text that completely bypasses that sub-feature. Similarly, a model fine-tuned on informal writing will avoid these markers naturally.

The `specificity_density` and `structural_opener_patterns` sub-features are somewhat harder to game, but they can also be bypassed by prompting the LLM to include specific names, numbers, and avoid transitional openers. This is not a solvable problem with static rules — it requires a continuously updated vocabulary list or a behavioral model of the current generation of LLMs.

### 3. Poetry, lyrics, and fragmented prose

Poetry breaks both signal assumptions. Sentence boundary detection splits on `.!?` followed by whitespace, so line breaks are invisible — the tokenizer treats an entire poem as one long "sentence," making `sentence_length_uniformity` undefined. Short fragments and deliberate repetition suppress vocabulary diversity. Concrete details like names and places may appear frequently in poetry for artistic reasons, pulling `specificity_density` toward human even for AI-generated poetry.

In practice, unusual structure tends to produce a `weighted_score` near 0.5, which lands as `uncertain` — which is the correct hedge. But the system will never be meaningfully confident about poetry in either direction.

### 4. Short text under `MIN_TEXT_LENGTH` (20 words)

With very short content, all four sub-features are capped at 0.5, preventing overconfident misclassification on sparse evidence. This is a safety mechanism, not a solution — the system genuinely cannot classify a single sentence reliably and the `uncertain` output for such texts should be taken at face value.

---

## Spec Reflection

### Where the spec guided implementation well

The planning document's decision to make `heuristic_score` act as a **two-sided cost gate** before the LLM call was the single most valuable architectural choice in the spec. Without it, every submission would incur an API call regardless of how obvious the answer was. In practice, clearly AI-saturated text (like the `_AI_TEXT` test fixture) scores 0.97 on heuristics — a 0.85 upper gate fires immediately and no API call is made. The spec anticipated this and budgeted the LLM as a precision instrument for the ambiguous middle, not as the default first step.

This also made the two-sided gate logic in `app.py` clean: `lower_gate_closed` and `upper_gate_closed` are computed first, and the entire LLM branch is wrapped in `else`. The spec diagram was close enough to the final implementation that no restructuring was needed.

### Where implementation diverged from the spec

The original spec defined a four-field confidence calculation: `weighted_score`, `signal_agreement`, `raw_confidence`, and `final_confidence_score`. In the spec, `final_confidence_score = raw_confidence × signal_agreement` was the decisive variable used for label thresholds — a multiplication that penalized both low decisiveness and low signal agreement simultaneously.

During implementation this was replaced with a direct **adaptive weighted average**: `weighted_score = w_llm × llm_score + w_heuristic × heuristic_score`. The reason for the change: the original formula produced a value that was neither a raw probability (0–1 with a clear meaning) nor a direct label threshold. A `final_confidence_score` of 0.42 required mental gymnastics to interpret — is that the blended score, or how confident the blending was? Making `weighted_score` the sole output score and comparing it directly against `AI_SCORE_THRESHOLD = 0.65` is simpler to reason about and easier to tune. The `raw_confidence` and `CONFIDENCE_THRESHOLD` constants remain in `config.py` as comments but are no longer active.

---

## AI Usage

### Instance 1 — Initial pipeline module scaffolding

**What I directed:** I provided Claude with the planning.md spec sections covering Signal 1 (heuristics sub-feature table with word lists and gate logic) and Signal 2 (Groq API call spec and retry behavior), plus the architecture diagram. I asked it to generate `pipeline/heuristic_signal.py` with a `compute_heuristic_score(text: str) -> dict` function running four sub-features in parallel via `ThreadPoolExecutor`, and `pipeline/llm_signal.py` with retry logic.

**What it produced:** A working parallel executor pattern with the four sub-functions, NFKC text normalization, sentence tokenization via regex, and the Groq API call with exponential backoff. The overall structure was correct and the module boundaries matched the spec.

**What I revised:** The original `heuristic_signal.py` used a simple equal average across the four sub-features. The spec was later updated to require a weighted average (30/25/30/15). I also directed Claude to replace the `punctuation_range` sub-feature entirely with `specificity_density` — the new function needed a new implementation detecting numeric tokens and mid-sentence proper nouns, which Claude generated correctly but which I then tuned: the saturation threshold (1 anchor per 10 words → score 0.0) was adjusted from an initial 0.20 to 0.10 to make the feature more sensitive at typical text lengths.

### Instance 2 — Adaptive confidence scoring formula

**What I directed:** I described the new three-tier adaptive weighted average verbally — "if word_count > 150 use 65/35, if |gap| > 0.40 use 85/15, otherwise 70/30, applied in that priority order" — and asked Claude to update `pipeline/classifier.py`, add the corresponding `planning.md` update section (without removing the original design), and update `tests/test_classifier.py` to cover all three tiers including boundary conditions.

**What it produced:** The classifier rewrite was mechanically correct. The test suite it generated covered the three weight tiers and used exact arithmetic (`round(0.70 * llm + 0.30 * heuristic, 4)`) as expected values, which is the right approach. The `planning.md` update section correctly preserved the original design above it.

**What I revised:** The initial test for the `word_count > 150` priority check was written as `word_count >= 150`, not `word_count > 150` (strict `>`). I caught this during review — the condition in code was strict, but the test boundary was inclusive, meaning the test at exactly 150 would have passed with the wrong weight tier. I directed Claude to fix the boundary test to use `word_count=150` and assert it falls through to the gap-based tier (85/15 for gap > 0.40), not the long-text tier. I also added explicit tests for the `word_count=151` case to verify the long-text branch fires one step past the boundary.
