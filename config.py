# Groq API retry settings
LLM_MAX_RETRIES = 3
LLM_RETRY_BASE_DELAY = 1.0  # seconds; each retry waits base * (2 ** attempt)

# Text length gate — heuristic sub-scores capped at 0.5 below this word count
MIN_TEXT_LENGTH = 80

# Cost gate — heuristic_score below this skips the LLM call entirely
HEURISTIC_GATE_THRESHOLD = 0.25

# Confidence thresholds for label assignment
CONFIDENCE_THRESHOLD = 0.70     # minimum final_confidence_score for a definitive label
AI_SCORE_THRESHOLD = 0.65       # weighted_score >= this → AI zone
HUMAN_SCORE_THRESHOLD = 0.35    # weighted_score <= this → human zone

# Applied to raw_confidence when llm_signal_available=False
SINGLE_SIGNAL_MULTIPLIER = 0.75

# Per-IP rate limit on POST /submit
RATE_LIMIT = "10 per minute"

# Flat-file persistence paths
SUBMISSIONS_FILE = "data/submissions.jsonl"
AUDIT_LOG_FILE = "data/audit_log.jsonl"
