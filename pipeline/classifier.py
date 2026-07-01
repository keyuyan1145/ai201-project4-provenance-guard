import config


def classify(heuristic_score: float, llm_score: float | None, word_count: int = 0) -> dict:
    """Combine heuristic and LLM signals into a single weighted_score (0–1).

    Dual-signal mode: adaptive weighted average. Weight tier is chosen in
    priority order:
        1. word_count > 150   → 65% LLM / 35% heuristic
        2. |gap| > 0.40       → 85% LLM / 15% heuristic
        3. otherwise          → 70% LLM / 30% heuristic

    weighted_score IS the final classification score — no separate
    confidence multiplier is applied. Label thresholds are compared directly
    against it.

    Single-signal mode (llm_score is None): weighted_score = heuristic_score.
    The gate handles definitive labels at the extremes; the middle range always
    yields "uncertain" because the LLM call was attempted but failed.
    """
    if llm_score is not None:
        gap = abs(llm_score - heuristic_score)
        signal_agreement = round(1.0 - gap, 4)

        if word_count > 150:
            w_llm, w_heuristic = 0.65, 0.35
        elif gap > 0.40:
            w_llm, w_heuristic = 0.85, 0.15
        else:
            w_llm, w_heuristic = 0.70, 0.30

        weighted_score = round(w_llm * llm_score + w_heuristic * heuristic_score, 4)
        llm_signal_available = True

        print(
            f"[INFO] Classifier: dual-signal (word_count={word_count}, gap={gap:.4f},"
            f" weights={w_llm}/{w_heuristic});"
            f" weighted_score={weighted_score:.4f}"
        )

    else:
        weighted_score = round(heuristic_score, 4)
        signal_agreement = None
        llm_signal_available = False

        print(
            f"[INFO] Classifier: single-signal (llm_signal_available=False);"
            f" weighted_score={weighted_score:.4f}"
        )

    return {
        "weighted_score": weighted_score,
        "signal_agreement": signal_agreement,
        "llm_signal_available": llm_signal_available,
    }
