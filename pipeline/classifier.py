import config


def classify(heuristic_score: float, llm_score: float | None) -> dict:
    """Combine heuristic and LLM signals into a final confidence score.

    When llm_score is None (single-signal mode), applies SINGLE_SIGNAL_MULTIPLIER
    as a confidence penalty to reflect reduced evidence.

    Returns a dict with: weighted_score, signal_agreement, raw_confidence,
    final_confidence_score, llm_signal_available.
    """
    if llm_score is not None:
        weighted_score = round(0.65 * llm_score + 0.35 * heuristic_score, 4)
        signal_agreement = round(1.0 - abs(llm_score - heuristic_score), 4)
        raw_confidence = round(2.0 * abs(weighted_score - 0.5), 4)
        final_confidence_score = round(raw_confidence * signal_agreement, 4)
        llm_signal_available = True
        print(
            f"[INFO] Classifier: dual-signal mode;"
            f" weighted_score={weighted_score:.4f},"
            f" signal_agreement={signal_agreement:.4f},"
            f" final_confidence_score={final_confidence_score:.4f}"
        )
    else:
        weighted_score = round(heuristic_score, 4)
        signal_agreement = None
        raw_confidence = round(2.0 * abs(weighted_score - 0.5), 4)
        final_confidence_score = round(raw_confidence * config.SINGLE_SIGNAL_MULTIPLIER, 4)
        llm_signal_available = False
        print(
            f"[INFO] Classifier: single-signal mode (llm_signal_available=False);"
            f" applying {config.SINGLE_SIGNAL_MULTIPLIER}x confidence penalty;"
            f" final_confidence_score={final_confidence_score:.4f}"
        )

    return {
        "weighted_score": weighted_score,
        "signal_agreement": signal_agreement,
        "raw_confidence": raw_confidence,
        "final_confidence_score": final_confidence_score,
        "llm_signal_available": llm_signal_available,
    }
