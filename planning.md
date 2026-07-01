# Provenance Guard — planning.md

A backend classification system that accepts text submissions, runs a multi-signal detection pipeline, returns a confidence-scored transparency label, and supports creator appeals. Built with Flask, Groq API, and flat-file (JSONL) persistence.

---

## 1. Detection Signals

> **Question: What are your 2+ signals? What does each one measure? What does each signal's output look like (a score between 0–1? a binary flag?), and how will you combine them into a single confidence score?**

### Signal 1 — Statistical Heuristics
**Runs first. Always executes. Acts as a cost gate.**

Four independent sub-features run in parallel via `ThreadPoolExecutor`, then combined via **weighted average** into a single `heuristic_score` (float, 0.0–1.0, where 1.0 = strongly AI-like).

| Sub-feature | Weight | What it measures | AI indicator |
|---|---|---|---|
| **AI Vocabulary Marker Density** | **30%** | Frequency of words/phrases statistically overrepresented in LLM output, normalized per 100 words. Words: `delve, certainly, straightforward, robust, seamless, nuanced, comprehensive, leverage, crucial, notably, invaluable, pivotal`. Phrases: `it is worth noting, in conclusion, it's important to, in today's, of course`. Starters: `Moreover, Furthermore, Additionally` | High density → AI |
| **Structural Opener Patterns** | **25%** | Fraction of sentences beginning with a transitional word: `However, Therefore, In addition, As a result, For example, In contrast, Overall, To summarize`. AI models over-use explicit logical signposting. | High fraction → AI |
| **Specificity and Concrete Details** | **30%** | Density of concrete anchors (numbers, percentages, mid-sentence proper nouns, named entities) per word count. AI text is abstract and generic; human text uses specific dates, figures, and names. Score = 1 − (anchor density / 0.10); saturates to 0.0 at 1 anchor per 10 words. | Low specificity → AI |
| **Sentence Length Uniformity** | **15%** | Standard deviation of per-sentence word counts, inverted and normalized. AI models produce rhythmically regular prose. | Low variance → AI |

**Weight rationale:** Weights were calibrated for the typical submission length of this system — short prose under 150 words. At that length, two of the four sub-features become unreliable:

- **Punctuation range** was removed entirely. Short texts rarely use more than two or three punctuation types regardless of authorship, so the signal is nearly always the same and carries no discriminative power. It was replaced with **Specificity and Concrete Details**, which detects the presence or absence of concrete anchors (numbers, names, dates) and works at any length.
- **Sentence length uniformity** receives the lowest weight (15%) because a short passage seldom contains more than four or five sentences. With that few data points, the coefficient of variation is noisy and a single outlier sentence can dominate the score.
- **Structural opener patterns** receives a reduced weight (25%) for the same reason: with few sentences, one opener-starting sentence inflates the fraction disproportionately. The signal is still useful but less reliable than in longer documents.
- **AI vocabulary density** and **Specificity and Concrete Details** each receive the highest weight (30%) because both operate at the word level — they pick up markers regardless of how many sentences are present and remain equally informative whether the text is 50 words or 500 words.

**Gate logic:** The heuristic score acts as a two-sided gate before Signal 2 runs:

| Condition | Action | Forced label |
|---|---|---|
| `heuristic_score < 0.15` | LLM **skipped** — strong human evidence across all surface features | `high_confidence_human` |
| `heuristic_score > 0.85` | LLM **skipped** — strong AI evidence across all surface features | `high_confidence_ai` |
| `0.15 ≤ heuristic_score ≤ 0.85` | LLM runs normally | Determined by dual-signal combination |

Both extremes bypass the API call entirely, eliminating cost where the heuristics already give a decisive answer.

**Output:** `heuristic_score` — float 0.0–1.0

**Blind spot:** Formal human writing (academic papers, legal briefs, structured essays) triggers multiple sub-features and will score elevated even when genuinely human. These cases are intentionally passed through the gate to Signal 2 for a deeper read.

---

### Signal 2 — LLM Semantic Classifier
**Runs second. Conditional on Signal 1 gate. Uses Groq API.**

The raw text is sent to a Llama model via Groq with a structured prompt asking it to assess the probability (0.0–1.0) that the text was AI-generated. The model returns a JSON object containing the score and brief reasoning. The call is retried up to a configurable number of times (default: 3) with exponential backoff.

**Fallback:** If all retries fail, the system sets `llm_signal_available: false`, falls back to heuristics-only mode, and forces the label to `"uncertain"` — a single signal is never sufficient for a definitive verdict.

**Output:** `llm_score` — float 0.0–1.0 (or `null` if unavailable)

**What it captures:** Holistic semantic patterns — whether rhetorical moves follow AI-typical patterns (claim → evidence → summary), whether vocabulary is competent but lacks personal idiosyncrasy, whether the text has the "too complete" quality of AI generation (every idea resolved, no loose threads).

**Blind spot:** Highly edited human writing and short texts (under ~100 words). Also inherits biases of the evaluating model — may be systematically better at detecting writing from the same model family.

---

### Combining Signals into a Confidence Score

Four computed fields feed into the final output:

| Field | Formula | What it measures |
|---|---|---|
| `weighted_score` | `0.65 × llm_score + 0.35 × heuristic_score` | Combined AI authorship verdict (0=human, 1=AI). LLM weighted higher because it reads semantics; heuristics are surface-level. When LLM unavailable: `weighted_score = heuristic_score`. |
| `signal_agreement` | `1 − \|llm_score − heuristic_score\|` | How much the two signals corroborate each other (1=perfect agreement, 0=completely opposite). High agreement means both methods tell the same story. |
| `raw_confidence` | `2 × \|weighted_score − 0.5\|` | How far the blended score is from the midpoint (0.5 = maximum uncertainty). The `2×` rescales to 0–1. Answers: *"How decisive is the combined score?"* |
| `final_confidence_score` | `raw_confidence × signal_agreement` | Overall certainty. Multiplication enforces that *both* conditions must hold: the score must be decisive AND the signals must agree. Either factor being weak pulls the whole confidence down. |

**Single-signal mode** (LLM skipped by gate or all retries failed): `final_confidence_score = raw_confidence × 0.75` — capped to reflect reduced evidence from one signal only.

---

## 2. Uncertainty Representation

> **Question: What does a confidence score of 0.6 mean to your system? How will you map raw signal outputs to a calibrated score? What threshold separates "likely AI" from "uncertain" from "likely human"?**

### What 0.6 confidence means

A `final_confidence_score` of 0.60 means the system has a moderate, not strong, basis for its verdict. Working backwards through the formula: a score of 0.60 could result from a `weighted_score` of 0.80 with `signal_agreement` of 0.75 (decisive score but signals only moderately agree), or from a `weighted_score` of 0.70 with near-perfect agreement (decisive but not extreme). In both cases, the system leans toward a verdict but isn't strongly confident. A 0.60 confidence falls **below the 0.70 threshold** required to trigger a strong label — the result is rendered as "uncertain."

### Thresholds

| Condition | Label variant |
|---|---|
| `weighted_score >= 0.65` AND `final_confidence_score >= 0.70` | `high_confidence_ai` |
| `weighted_score <= 0.35` AND `final_confidence_score >= 0.70` | `high_confidence_human` |
| Everything else | `uncertain` |

The 0.70 confidence threshold is a deliberate conservative choice. Setting it lower increases the risk of damning correct verdicts on borderline cases; 0.70 means both the score and signal agreement must be meaningfully strong before a definitive label is issued.

The 0.65 / 0.35 weighted_score thresholds define a symmetric "uncertain zone" around the midpoint. Content scoring between 0.35 and 0.65 is treated as genuinely ambiguous regardless of confidence.

### Score mapping

Raw signal outputs (both `llm_score` and each heuristic sub-score) are already normalized to 0–1 at the point they are produced. No additional calibration step is applied. The confidence formula (`raw_confidence × signal_agreement`) naturally compresses extreme-but-conflicting cases and expands cases where both signals align and are decisive.

---

## 3. Transparency Label Design

> **Question: What exact text will the label show for a high-confidence AI result? A high-confidence human result? An uncertain result? Write out the three label variants now, before you build the UI.**

The `label` field is a single plain-text string returned in the API response. The platform renders it verbatim. The confidence percentage is interpolated at generation time.

### Variant 1 — `high_confidence_ai`
*Triggered when: `weighted_score >= 0.65` AND `final_confidence_score >= 0.70`*

> "This content shows strong indicators of AI authorship. Our system analyzed the text across multiple signals and found patterns consistent with AI-generated writing (confidence: {X}%). If you are the creator and believe this is incorrect, you can submit an appeal using your submission ID."

### Variant 2 — `high_confidence_human`
*Triggered when: `weighted_score <= 0.35` AND `final_confidence_score >= 0.70`*

> "This content appears to be human-written. Our system analyzed the text across multiple signals and found no significant indicators of AI authorship (confidence: {X}%)."

### Variant 3 — `uncertain`
*Triggered when: everything else (including LLM fallback mode)*

> "The origin of this content is unclear. Our system detected mixed or inconclusive signals and cannot make a confident determination (confidence: {X}%). This result should be interpreted with caution. If you are the creator of this content and believe it is human-written, you may submit an appeal using your submission ID."

**LLM fallback addition** (appended when `llm_signal_available: false`):

> " Note: the AI signal was temporarily unavailable; this result is based on surface-level analysis only."

---

## 4. Appeals Workflow

> **Question: Who can submit an appeal? What information do they provide? What does the system do when an appeal is received — what status changes, what gets logged? What would a human reviewer see when they open the appeal queue?**

### Who can submit

Any creator who received a `label_id` (submission UUID) in a prior `POST /submit` response. No authentication is enforced by this system — the platform is responsible for ensuring only the original creator submits an appeal for their content.

### What they provide

```json
{
  "submission_id": "uuid-string",
  "reasoning": "free-text explanation — why the creator believes the classification is wrong"
}
```

The appeals path is intentionally frictionless: one reference ID and free text. No file attachments, no structured form fields. Creating friction punishes legitimate human writers.

### What the system does

1. Validates `submission_id` exists in `data/submissions.jsonl`. Returns `404` if not found.
2. Updates the submission record's `status` field from `"classified"` to `"under_review"` (full record rewrite in-place).
3. Generates a new `appeal_id` UUID.
4. Appends an entry to `data/audit_log.jsonl`:
   ```json
   {
     "event_type": "appeal",
     "appeal_id": "uuid",
     "submission_id": "uuid",
     "reasoning": "creator's text",
     "created_at": "ISO8601 timestamp"
   }
   ```
5. Returns confirmation response with `appeal_id`, `submission_id`, `status: "under_review"`, and a plain-language message.

No automated re-classification occurs. The appeal is a flag for human review.

### What a human reviewer sees (via `GET /log`)

The reviewer queries `GET /log?event_type=appeal` and sees each appeal entry. By cross-referencing the `submission_id` against `GET /log?event_type=classification`, they can reconstruct the full picture:

- Original `llm_score` and `heuristic_score`
- `weighted_score` and `final_confidence_score` at time of classification
- The label that was shown to the user
- The creator's written reasoning for the appeal
- Timestamps for both the original classification and the appeal

---

## 5. Anticipated Edge Cases

> **Question: What types of content will your system handle poorly? Name at least two specific scenarios — not generic risks like "inaccurate detection," but specific cases.**

### Edge case 1 — Formal academic writing by a human
A graduate student writes a structured 600-word essay with uniform paragraph lengths, transitional phrases ("Furthermore", "In conclusion"), and precise but unsurprising vocabulary. Signal 1 heuristics will flag the text: low sentence variance, elevated opener density, and AI vocabulary markers all trigger. If Signal 2 also finds the text "too complete", the system may issue a high-confidence AI verdict for entirely human work.

**Mitigation:** The 0.70 confidence threshold and the `signal_agreement` penalty reduce the risk. If the signals score differently (heuristic high, LLM moderate), `signal_agreement` pulls `final_confidence_score` down and the result falls into "uncertain" rather than "high_confidence_ai". The appeals workflow exists as the safety valve for the cases that still slip through.

### Edge case 2 — A poem or song lyrics
Poetry breaks the assumptions of both signals. Sentence boundaries are ambiguous (line breaks ≠ sentences), vocabulary is intentionally constrained or repetitive, and structural signposting is absent. Heuristics will produce unreliable sub-scores: sentence length variance is undefined for fragments, punctuation range is intentionally narrow, and TTR may be artificially low due to deliberate repetition. The LLM signal is also less reliable here — it may misread poetic compression as AI uniformity or vice versa.

**Mitigation:** Short or structurally unusual texts tend to produce `weighted_score` values near 0.5 (the midpoint), which means `raw_confidence` will be low, and the result will land in "uncertain". This is the correct behavior — the system should hedge, not guess.

### Edge case 3 — AI-generated text that was heavily edited by a human
A creator generates a first draft with an AI tool, then spends two hours rewriting sentences, injecting personal details, and varying rhythm. The surface heuristics may no longer apply (editing removes the obvious markers), but the LLM signal may still detect the underlying semantic structure. The two signals will likely disagree, producing low `signal_agreement` and therefore low `final_confidence_score` — correctly landing as "uncertain".

### Edge case 4 — Very short content (under MIN_TEXT_LENGTH words)
A haiku, a tweet-length post, or a brief product description provides insufficient material for meaningful heuristic computation. Standard deviation of sentence lengths across 2–3 sentences is statistically noisy; vocabulary diversity metrics are unreliable at small sample sizes. The LLM signal is more robust at short lengths but still less reliable than on longer texts.

**Mitigation:** Heuristic sub-scores are capped at 0.5 for texts under `MIN_TEXT_LENGTH` (configurable in `config.py`, default: 80 words), preventing confident misclassification on sparse evidence. The short-text condition is logged with `print()` so it is visible during debugging.

---

## Architecture

> **Include the diagram you drew in Milestone 1 and a 2–3 sentence narrative describing the submission and appeal flows.**

### Submission Flow

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
                      ┌──────────────────────────┐
                      │    Classifier            │
                      │  weighted_score,         │
                      │  signal_agreement,       │
                      │  raw_confidence,         │
                      │  final_confidence_score  │
                      └────────┬─────────────────┘
                               │ weighted_score: float, final_confidence_score: float
                               ▼
                      ┌──────────────────┐
                      │  Label Generator │
                      │  (3 variants)    │
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
                      │  {label_id, content_id, weighted_score,               │
                      │   final_confidence_score, attribution, label,         │
                      │   llm_score, heuristic_score, agreement_score}        │
                      └─────────────────────────────────────────────────────┘
```

### Appeal Flow

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

### Narrative

On submission, a piece of text passes through a rate limiter and validator before entering a two-stage sequential detection pipeline: the heuristic signal always runs first (cheap, local, no API cost) and acts as a gate — only if the text scores ambiguous or AI-leaning does it proceed to the LLM classifier. The classifier combines whichever signals ran into a `weighted_score` and `final_confidence_score`, the label generator maps these to one of three human-readable variants, and the result is persisted to flat JSONL files before the response is returned.

On appeal, the creator references their `label_id` (returned in the original response) with a free-text explanation; the system updates the submission status to `"under_review"` and appends a structured entry to the audit log. No automated re-classification occurs — the audit log is the paper trail a human reviewer reads to make a final determination.

---

## AI Tool Plan

### M3 — Submission Endpoint + Signal 1 (Heuristics)

**Spec sections to provide:**
- Section 1 (Detection Signals) — Signal 1 sub-feature table with word lists, gate logic, and `heuristic_score` output definition
- Architecture section — Submission flow diagram, specifically the nodes from Request Validator through Gate Check

**What to ask the AI tool to generate:**
1. Flask app skeleton: `app.py` with the `POST /submit` route, request body validation (`content` + `creator_id` required), UUID generation for `submission_id`, and flask-limiter setup (10 requests/minute per IP)
2. `pipeline/heuristic_signal.py`: a single `compute_heuristic_score(text: str) -> float` function that runs the four sub-features in parallel via `ThreadPoolExecutor` and returns their average — each sub-feature as its own function returning a 0–1 float

**How to verify before wiring in:**
Call `compute_heuristic_score()` directly on at least three test strings:
- Obvious AI text (copy a paragraph from a ChatGPT response) — expect score > 0.60
- Casual human text (a personal blog post or informal email excerpt) — expect score < 0.35
- Borderline text (a structured student essay) — expect score in 0.35–0.65 range

Check that the four sub-feature scores are individually inspectable (return a dict, not just the average) so scoring behavior can be debugged. Only wire into the endpoint after scores feel calibrated.

**Unit tests (`tests/test_heuristic_signal.py`):**
- Each sub-feature function tested in isolation with a hand-crafted input that should produce a known high or low score (e.g., a string with five "Moreover" starters → high opener score)
- `compute_heuristic_score()` always returns a float in [0.0, 1.0]
- `POST /submit` returns `400` when `content` is missing
- `POST /submit` returns `400` when `creator_id` is missing
- `POST /submit` with valid body returns `200` with all six expected response fields present

---

### M4 — Signal 2 + Confidence Scoring

**Spec sections to provide:**
- Section 1 (Detection Signals) — Signal 2 description, Groq API call spec, retry/fallback behavior, and the four-field combining formula table (`weighted_score`, `signal_agreement`, `raw_confidence`, `final_confidence_score`)
- Section 2 (Uncertainty Representation) — threshold table and the worked example of what 0.60 confidence means
- Architecture section — Gate Check → Signal 2 → Classifier nodes from the submission flow diagram

**What to ask the AI tool to generate:**
1. `pipeline/llm_signal.py`: a `compute_llm_score(text: str) -> float | None` function that calls Groq with a structured prompt, parses the returned probability, retries up to a configurable max (default 3) with exponential backoff, and returns `None` on total failure
2. `pipeline/classifier.py`: a `classify(heuristic_score: float, llm_score: float | None) -> dict` function that computes all four fields and returns them as a dictionary; handles single-signal mode (LLM `None`) by using the capped formula `raw_confidence × 0.75`

**What to check:**
- Run the full pipeline on clearly AI-generated text vs clearly human text and confirm `weighted_score` differs by at least 0.30 between them
- Run on a borderline case and confirm `final_confidence_score` is below 0.70 (lands as uncertain)
- Simulate LLM failure (pass `llm_score=None`) and confirm `final_confidence_score` is appropriately capped and the result would produce an "uncertain" label
- Confirm the gate works end-to-end: submit text that scores < 0.25 on heuristics and verify `llm_score` is `null` in the response

**Unit tests (`tests/test_classifier.py`):**
- `classify(heuristic_score=0.80, llm_score=0.85)` → `weighted_score` ≈ 0.8175, `signal_agreement` ≈ 0.95, `final_confidence_score` > 0.70
- `classify(heuristic_score=0.50, llm_score=0.55)` → `final_confidence_score` < 0.20 (near-midpoint score = low confidence)
- `classify(heuristic_score=0.80, llm_score=0.20)` → low `signal_agreement` pulls `final_confidence_score` below 0.70 despite decisive individual scores
- `classify(heuristic_score=0.70, llm_score=None)` → single-signal mode: `final_confidence_score = raw_confidence × 0.75`, `llm_score` absent from output
- All output values are floats clamped to [0.0, 1.0]

---

### M5 — Production Layer (Labels, Appeals, Audit Log)

**Spec sections to provide:**
- Section 3 (Transparency Label Design) — all three variant trigger conditions and exact label text strings including the LLM fallback addendum
- Section 4 (Appeals Workflow) — request shape, the five system steps, and the audit log entry schema
- Architecture section — full submission flow diagram (Label Generator → File Write → Response) and the complete appeal flow diagram

**What to ask the AI tool to generate:**
1. `labels.py`: a `generate_label(weighted_score: float, final_confidence_score: float, llm_signal_available: bool) -> str` function that maps scores to one of the three variants with the confidence percentage interpolated, appending the fallback note when LLM was unavailable
2. `POST /appeals` route: validates `submission_id` + `reasoning`, scans `data/submissions.jsonl` for the record, rewrites it with `status: "under_review"`, appends an appeal entry to `data/audit_log.jsonl`, returns the confirmation response
3. `GET /log` route: reads `data/audit_log.jsonl`, supports optional `?limit` and `?event_type` query params, returns paginated entries

**How to verify:**

| Test | Expected result |
|---|---|
| Submit clear AI text → check `label` field | `high_confidence_ai` variant text |
| Submit clear human text → check `label` field | `high_confidence_human` variant text |
| Submit borderline text → check `label` field | `uncertain` variant text |
| Mock LLM unavailable → check `label` field | `uncertain` + fallback note appended |
| `POST /appeals` with valid `submission_id` → re-read `submissions.jsonl` | Record status changed to `"under_review"` |
| `POST /appeals` with valid `submission_id` → `GET /log?event_type=appeal` | Appeal entry present with matching `submission_id` and `reasoning` |
| `POST /appeals` with unknown `submission_id` | `404` response |
| `GET /log` after 3+ submissions | At least 3 classification entries visible |

**Unit tests (`tests/test_labels.py`, `tests/test_appeals.py`):**
- `generate_label(weighted_score=0.80, final_confidence_score=0.75, llm_signal_available=True)` → string contains "strong indicators of AI authorship"
- `generate_label(weighted_score=0.20, final_confidence_score=0.80, llm_signal_available=True)` → string contains "human-written"
- `generate_label(weighted_score=0.60, final_confidence_score=0.50, llm_signal_available=True)` → string contains "unclear"
- `generate_label(weighted_score=0.80, final_confidence_score=0.75, llm_signal_available=False)` → string contains both "strong indicators of AI authorship" and the fallback note
- Confidence percentage in label text matches `round(final_confidence_score * 100)` for each variant
- `POST /appeals` with valid `submission_id` → response `status` field equals `"under_review"`
- `POST /appeals` with valid `submission_id` → `GET /log?event_type=appeal` returns entry with correct `submission_id` and `reasoning`
- `POST /appeals` with missing `reasoning` field → `400` response
- `GET /log?limit=2` → returns at most 2 entries
- `GET /log?event_type=classification` → returns only classification entries, no appeal entries

---

## LLM System Prompt

Used by `pipeline/llm_signal.py` when calling Groq. The prompt is sent as the `system` role; the submitted text is sent as the `user` role.

### Prompt design principles applied
- **Clear role**: establishes a specific expert persona so the model stays in analysis mode
- **Explicit dimensions**: named criteria prevent the model from inventing its own rubric
- **Hard output constraint**: "ONLY a valid JSON object" with no text outside reduces parsing failures
- **Typed schema with field definitions**: inline descriptions so the model knows exactly what each key means
- **Few-shot examples**: one clearly-AI and one clearly-human example anchor the scoring scale at both ends
- **Specific vocabulary in examples**: mirrors the heuristic word lists, reinforcing consistency between signals

```
You are an expert forensic linguist specializing in AI-generated text detection.
Your task is to analyze a piece of writing and estimate the probability that it
was generated by a large language model (LLM) rather than written by a human.

Analyze the following dimensions:
- RHETORICAL STRUCTURE: Does the text follow a predictable AI pattern
  (claim → evidence → summary)? Does every idea feel resolved with no loose
  threads, tangents, or raw emotion?
- VOCABULARY: Is the vocabulary competent but lacking personal idiosyncrasy?
  Does it over-use words like "delve", "robust", "nuanced", "leverage",
  "comprehensive", "crucial", "seamless"?
- REGISTER CONSISTENCY: Is the tone perfectly uniform throughout, or does it
  have the natural shifts and seams of human writing (fatigue, escalation,
  digression)?
- HEDGING LANGUAGE: Does hedging appear in AI-characteristic ways —
  over-qualified statements, systematic caveats applied uniformly?
- SIGNPOSTING: Are transitional phrases (Moreover, Furthermore, In conclusion,
  It is worth noting) used at a density typical of LLM output?

You MUST respond ONLY with a valid JSON object. Do not include any text,
explanation, or markdown outside the JSON object.

Output schema:
{
  "ai_probability": <float between 0.0 and 1.0>,
  "reasoning": "<1-2 sentences identifying the most decisive signals>",
  "key_signals": ["<signal 1>", "<signal 2>", "<signal 3>"]
}

Field definitions:
- ai_probability: 0.0 = certainly human-written, 1.0 = certainly AI-generated.
  Use the full range — do not cluster near 0.5 unless genuinely uncertain.
- reasoning: the 1-2 most decisive features that drove your score, written for
  a technical reviewer. Name specific observations, not generic categories.
- key_signals: exactly 2-3 specific observed textual features.
  Examples: "high AI vocabulary marker density", "uniform sentence rhythm",
  "absence of personal voice or idiosyncrasy", "overuse of transitional openers"

---

EXAMPLE 1 — Clearly AI-generated:

User: "Delving into the realm of machine learning, it is worth noting that
neural networks have revolutionized many industries. Moreover, the robust
framework provides a comprehensive solution to previously intractable problems.
In conclusion, leveraging these technologies is crucial for modern enterprises."

Assistant: {
  "ai_probability": 0.95,
  "reasoning": "Multiple high-frequency AI vocabulary markers ('delve', 'robust', 'comprehensive', 'leverage', 'crucial') appear in only 4 sentences, and the text follows the exact claim-to-evidence-to-summary rhetorical pattern typical of LLM output.",
  "key_signals": ["high AI vocabulary marker density", "heavy transitional opener usage (Moreover, In conclusion)", "perfectly resolved rhetorical structure with no loose threads"]
}

EXAMPLE 2 — Clearly human-written:

User: "been thinking about the ML stuff we talked about last week. honestly still not sure neural nets are the right call for our use case — we've got like maybe 200 labeled examples total? tried a basic logistic regression first (took 20 mins, worked ok) but my manager keeps pushing for 'something fancier'. idk. will probably just run both and see what blows up first"

Assistant: {
  "ai_probability": 0.04,
  "reasoning": "Strongly human-authored: stream-of-consciousness structure with unresolved thoughts, informal register with contractions and hedging ('honestly', 'idk'), specific situational detail (200 examples, 20 minutes), and no AI-characteristic signposting or vocabulary.",
  "key_signals": ["informal register with natural hedging and uncertainty", "absence of transitional signposting", "unresolved ideas and raw pragmatic concern (manager pressure)"]
}
```

---

## Audit Log Schema

File: `data/audit_log.jsonl` — one JSON object per line, two event types.

### Classification entry

Written immediately after every successful `POST /submit` response.

```json
{
  "event_type": "classification",
  "content_id": "uuid-string",
  "creator_id": "string",
  "timestamp": "2024-01-15T10:30:00.123Z",
  "attribution": "uncertain",
  "confidence": 0.487,
  "heuristic_score": 0.72,
  "llm_score": 0.78,
  "agreement_score": 0.94,
  "status": "classified"
}
```

**Field notes:**
- `content_id` is the same UUID returned to the user as `label_id` / `content_id` in the API response — the link between the API receipt and the audit log entry
- `creator_id` is stored for traceability during appeals; it is not returned in the API response
- `attribution` is the label variant assigned: `high_confidence_ai`, `high_confidence_human`, or `uncertain`
- `confidence` is `final_confidence_score` — the calibrated certainty value after signal weighting
- `agreement_score` is `1 − |llm_score − heuristic_score|` — how closely the two signals corroborate each other; `null` in single-signal mode (LLM skipped or unavailable)
- `llm_score` is `null` when the LLM was skipped by the cost gate or all retries failed
- `timestamp` is ISO 8601 UTC with millisecond precision: `YYYY-MM-DDTHH:MM:SS.mmmZ`

### Appeal entry

Written when `POST /appeals` is successfully processed.

```json
{
  "event_type": "appeal",
  "appeal_id": "uuid-string",
  "submission_id": "uuid-string",
  "reasoning": "I wrote this essay myself for my English class...",
  "created_at": "2024-01-15T11:45:00Z"
}
```

**Field notes:**
- `submission_id` links back to the original classification entry — a reviewer queries both to reconstruct the full picture
- `reasoning` is raw creator text, untouched — no sanitization beyond length validation
- No original scores or label are duplicated here; reviewers cross-reference the classification entry by `submission_id`

### Reviewer query pattern

A human reviewer reads `GET /log?event_type=appeal` to find pending appeals, then for each one reads `GET /log?event_type=classification` and matches on `submission_id` to pull the original scores and label. No join logic is needed in the API — the consumer handles it.

---

## Configuration (config.py)

All tuneable values live in `config.py` at the project root. No values are hardcoded in pipeline logic — they import from this module.

```python
# config.py

# Groq API retry settings
LLM_MAX_RETRIES = 3          # number of retry attempts before falling back to heuristics-only
LLM_RETRY_BASE_DELAY = 1.0   # seconds; each retry waits base * (2 ** attempt)

# Text length gate
MIN_TEXT_LENGTH = 80         # word count below which heuristic sub-scores are capped at 0.5
                             # prevents confident misclassification on sparse evidence

# Heuristic cost gate
HEURISTIC_GATE_THRESHOLD = 0.25  # heuristic_score below this skips the LLM call entirely

# Confidence thresholds for label assignment
CONFIDENCE_THRESHOLD = 0.70      # minimum final_confidence_score for a definitive label
AI_SCORE_THRESHOLD = 0.65        # weighted_score at or above this is the AI zone
HUMAN_SCORE_THRESHOLD = 0.35     # weighted_score at or below this is the human zone

# Single-signal confidence penalty
SINGLE_SIGNAL_MULTIPLIER = 0.75  # applied to raw_confidence when llm_signal_available=False

# Rate limiting
RATE_LIMIT = '10 per minute'     # per-IP limit on POST /submit

# JSONL file paths
SUBMISSIONS_FILE = 'data/submissions.jsonl'
AUDIT_LOG_FILE = 'data/audit_log.jsonl'
```

**Why config.py over environment variables:** These are operational tuning parameters, not secrets. They benefit from being visible in source control (code review, history) and importable by unit tests without env setup. Secrets (API keys) stay in `.env`.

---

## Logging Strategy

All logging uses `print()` statements (no logging framework). Each print includes enough context to reconstruct what happened without reading surrounding code.

### Submission pipeline — print() locations

| Location | Condition | Print message |
|---|---|---|
| Request validator | Missing `content` field | `[WARN] POST /submit: missing 'content' field` |
| Request validator | Missing `creator_id` field | `[WARN] POST /submit: missing 'creator_id' field` |
| Heuristic signal | Text below `MIN_TEXT_LENGTH` | `[INFO] Short text detected ({word_count} words < {MIN_TEXT_LENGTH}): sub-scores capped at 0.5` |
| Heuristic signal | Sub-feature returns out-of-range value | `[WARN] Heuristic sub-feature '{name}' returned {val} outside [0,1]; clamping` |
| Gate check | LLM call skipped | `[INFO] Gate: heuristic_score={score:.3f} < {HEURISTIC_GATE_THRESHOLD} — LLM call skipped` |
| LLM signal | Call attempt | `[INFO] LLM call attempt {attempt}/{LLM_MAX_RETRIES} for submission {submission_id}` |
| LLM signal | Non-200 response | `[WARN] LLM attempt {attempt} failed: HTTP {status_code}` |
| LLM signal | JSON parse failure | `[WARN] LLM attempt {attempt}: failed to parse response as JSON` |
| LLM signal | Missing `ai_probability` key | `[WARN] LLM attempt {attempt}: response JSON missing 'ai_probability' key` |
| LLM signal | `ai_probability` out of range | `[WARN] LLM attempt {attempt}: ai_probability={val} out of [0,1] range; clamping` |
| LLM signal | All retries exhausted | `[ERROR] LLM all {LLM_MAX_RETRIES} attempts failed for {submission_id} — falling back to heuristics-only` |
| Classifier | Single-signal mode active | `[INFO] Classifier: single-signal mode (llm_signal_available=False); applying {SINGLE_SIGNAL_MULTIPLIER}x confidence penalty` |
| Label generator | Label assigned | `[INFO] Label assigned: variant={variant}, weighted_score={ws:.3f}, final_confidence={fc:.3f}` |
| File write | Submissions file write | `[INFO] Submission {submission_id} persisted to {SUBMISSIONS_FILE}` |
| File write | Audit log write | `[INFO] Audit entry written: event=classification, submission_id={submission_id}` |
| File write | Write error | `[ERROR] Failed to write to {filepath}: {exc}` |

### Appeals pipeline — print() locations

| Location | Condition | Print message |
|---|---|---|
| Request validator | Missing `submission_id` | `[WARN] POST /appeals: missing 'submission_id' field` |
| Request validator | Missing `reasoning` | `[WARN] POST /appeals: missing 'reasoning' field` |
| File lookup | `submission_id` not found | `[WARN] Appeal: submission_id={sid} not found in {SUBMISSIONS_FILE}` |
| Status updater | Status update complete | `[INFO] Submission {sid} status updated to 'under_review'` |
| Audit logger | Appeal entry written | `[INFO] Audit entry written: event=appeal, appeal_id={aid}, submission_id={sid}` |

### GET /log — print() locations

| Location | Condition | Print message |
|---|---|---|
| File read | Audit log file not found | `[WARN] GET /log: {AUDIT_LOG_FILE} does not exist — returning empty result` |
| File read | Malformed JSON line | `[WARN] GET /log: skipping malformed line {line_num} in {AUDIT_LOG_FILE}` |

### Format convention

All print statements use `[LEVEL]` prefix: `[INFO]` for normal flow milestones, `[WARN]` for recoverable anomalies, `[ERROR]` for failures that degrade behavior. Variable values are interpolated inline so each line is self-contained in log output.
