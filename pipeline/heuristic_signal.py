import re
import statistics
import unicodedata
from concurrent.futures import ThreadPoolExecutor

import config

# ---------------------------------------------------------------------------
# Marker lists (from planning.md Signal 1 spec)
# ---------------------------------------------------------------------------

_AI_WORDS = {
    # Classic GPT/LLM vocabulary markers
    "delve", "certainly", "straightforward", "robust", "seamless", "nuanced",
    "comprehensive", "leverage", "crucial", "notably", "invaluable", "pivotal",
    # Extended markers seen frequently in AI-generated prose
    "essential", "transformative", "ethical", "ensure", "responsible",
    "stakeholders", "innovative", "facilitate", "utilize", "demonstrate",
    "paramount", "imperative", "underscore", "encompass", "foster",
}

_AI_PHRASES = [
    "it is worth noting",
    "it is important to",      # catches both "it is important to note" and "it's" form
    "it's important to",
    "it is essential to",
    "it is crucial to",
    "in conclusion",
    "in today's",
    "of course",
    "ensure that",
    "it is worth emphasizing",
]

# Checked at sentence-start position only (punctuation stripped before match)
_AI_STARTERS = {"moreover", "furthermore", "additionally", "importantly"}

# Full sentence openers — also includes the _AI_STARTERS so structural_openers
# sub-score fires on "Furthermore,..." style sentences
_STRUCTURAL_OPENERS = {
    "however",
    "therefore",
    "moreover",
    "furthermore",
    "additionally",
    "importantly",
    "in addition",
    "as a result",
    "for example",
    "in contrast",
    "overall",
    "to summarize",
}


def _normalize_text(text: str) -> str:
    """Normalize text before heuristic scoring.

    Applies NFKC unicode normalization (resolves ligatures and compatibility
    characters), normalizes line endings, and collapses runs of whitespace to
    a single space so all sub-features operate on a clean, consistent string.
    """
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokenize_sentences(text: str) -> list:
    """Split on . ! ? followed by whitespace. Good enough for paragraph-length prose."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in parts if s.strip()]


# ---------------------------------------------------------------------------
# Sub-features (each returns float in [0.0, 1.0])
# ---------------------------------------------------------------------------

def _score_vocab_marker_density(text: str) -> float:
    """Density of AI vocabulary markers per 100 words, normalized to [0, 1]."""
    words = text.split()
    if not words:
        return 0.0

    cleaned_words = [re.sub(r"[^a-z]", "", w.lower()) for w in words]
    word_hits = sum(1 for w in cleaned_words if w in _AI_WORDS)

    text_lower = text.lower()
    phrase_hits = sum(text_lower.count(p) for p in _AI_PHRASES)

    sentences = _tokenize_sentences(text)
    starter_hits = 0
    for sentence in sentences:
        parts = sentence.lower().strip().split()
        if parts:
            first_word = re.sub(r"[^a-z]", "", parts[0])
            if first_word in _AI_STARTERS:
                starter_hits += 1

    total = word_hits + phrase_hits + starter_hits
    density = (total / len(words)) * 100  # markers per 100 words
    return round(min(1.0, density / 5.0), 4)  # 5+ per 100 words saturates at 1.0


def _score_sentence_length_uniformity(text: str) -> float:
    """Low coefficient of variation in sentence word-counts → high score (AI-like)."""
    sentences = _tokenize_sentences(text)
    if len(sentences) < 2:
        return 0.5  # insufficient data

    lengths = [len(s.split()) for s in sentences]
    try:
        std = statistics.stdev(lengths)
        mean = statistics.mean(lengths)
    except statistics.StatisticsError:
        return 0.5

    if mean == 0:
        return 0.5

    cv = std / mean  # 0 = perfectly uniform, higher = more varied
    # CV=0 → score=1.0;  CV≥0.5 → score=0.0
    return round(max(0.0, 1.0 - (cv / 0.5)), 4)


def _score_specificity_density(text: str) -> float:
    """Low specificity (no concrete details) → high score (AI-like).

    AI text is abstract and generic. Human text contains concrete anchors:
    numbers, percentages, proper nouns, and named entities.
    Density = (numeric tokens + mid-sentence proper nouns) / word count.
    Saturates at 0.10 (1 specific anchor per 10 words) → score 0.0.
    """
    words = text.split()
    if not words:
        return 0.5

    numeric_hits = len(re.findall(r'\b\d+\.?\d*\s*%?\b', text))

    sentences = _tokenize_sentences(text)
    proper_noun_hits = 0
    for sent in sentences:
        parts = sent.split()
        for word in parts[1:]:  # skip sentence-initial capitalized word
            clean = re.sub(r"[^a-zA-Z]", "", word)
            if len(clean) > 1 and clean[0].isupper() and clean.lower() not in {"i", "ai"}:
                proper_noun_hits += 1

    density = (numeric_hits + proper_noun_hits) / len(words)
    # density=0 → 1.0 (no specifics → AI-like); density≥0.10 → 0.0 (rich specifics → human-like)
    return round(max(0.0, 1.0 - (density / 0.10)), 4)


def _score_structural_openers(text: str) -> float:
    """Fraction of sentences beginning with a structural signpost word."""
    sentences = _tokenize_sentences(text)
    if not sentences:
        return 0.0

    hits = 0
    for sentence in sentences:
        s_lower = sentence.lower().strip()
        for opener in _STRUCTURAL_OPENERS:
            if s_lower.startswith(opener):
                hits += 1
                break

    return round(hits / len(sentences), 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_SUB_FEATURE_WEIGHTS = {
    "vocab_marker_density": 0.30,
    "structural_opener_patterns": 0.25,
    "specificity_density": 0.30,
    "sentence_length_uniformity": 0.15,
}


def compute_heuristic_score(text: str) -> dict:
    """
    Run all four sub-features in parallel, combine via weighted average, and
    return a results dict.

    Sub-feature weights:
        vocab_marker_density       30%
        structural_opener_patterns 25%
        specificity_density        30%
        sentence_length_uniformity 15%

    Returns:
        {
            "heuristic_score": float,      # weighted sum of four sub-scores
            "sub_scores": {                # individual feature scores
                "vocab_marker_density": float,
                "structural_opener_patterns": float,
                "specificity_density": float,
                "sentence_length_uniformity": float,
            },
            "word_count": int,
            "is_short_text": bool,         # True when below MIN_TEXT_LENGTH
        }
    """
    text = _normalize_text(text)
    word_count = len(text.split())
    is_short = word_count < config.MIN_TEXT_LENGTH

    if is_short:
        print(
            f"[INFO] Short text detected ({word_count} words < {config.MIN_TEXT_LENGTH}):"
            " sub-scores capped at 0.5"
        )

    sub_features = [
        ("vocab_marker_density", _score_vocab_marker_density),
        ("structural_opener_patterns", _score_structural_openers),
        ("specificity_density", _score_specificity_density),
        ("sentence_length_uniformity", _score_sentence_length_uniformity),
    ]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {name: executor.submit(fn, text) for name, fn in sub_features}
        sub_scores = {name: future.result() for name, future in futures.items()}

    if is_short:
        sub_scores = {k: min(v, 0.5) for k, v in sub_scores.items()}

    for name, val in list(sub_scores.items()):
        if not (0.0 <= val <= 1.0):
            print(f"[WARN] Heuristic sub-feature '{name}' returned {val} outside [0,1]; clamping")
            sub_scores[name] = max(0.0, min(1.0, val))

    print(
        f"[DEBUG] Sub-scores:"
        f" vocab={sub_scores['vocab_marker_density']:.4f},"
        f" struct={sub_scores['structural_opener_patterns']:.4f},"
        f" specificity={sub_scores['specificity_density']:.4f},"
        f" uniformity={sub_scores['sentence_length_uniformity']:.4f}"
    )

    heuristic_score = round(
        sum(_SUB_FEATURE_WEIGHTS[k] * sub_scores[k] for k in _SUB_FEATURE_WEIGHTS),
        4,
    )
    print(f"[DEBUG] heuristic_score={heuristic_score:.4f} (weights 30/25/30/15)")

    return {
        "heuristic_score": heuristic_score,
        "sub_scores": sub_scores,
        "word_count": word_count,
        "is_short_text": is_short,
    }
