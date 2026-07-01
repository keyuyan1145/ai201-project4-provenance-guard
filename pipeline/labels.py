def generate_label_text(
    variant: str,
    confidence: float,
    llm_failure: bool = False,
) -> str:
    """Return the full human-readable transparency label string.

    Args:
        variant:     One of "high_confidence_ai", "high_confidence_human", "uncertain".
        confidence:  final_confidence_score (float 0-1); interpolated as a percentage.
        llm_failure: True when the LLM was attempted but all retries failed.
                     Appends a caveat note; NOT set when the gate simply skipped the LLM.
    """
    pct = round(confidence * 100)

    if variant == "high_confidence_ai":
        text = (
            f"This content shows strong indicators of AI authorship. Our system analyzed "
            f"the text across multiple signals and found patterns consistent with "
            f"AI-generated writing (confidence: {pct}%). If you are the creator and "
            f"believe this is incorrect, you can submit an appeal using your submission ID."
        )
    elif variant == "high_confidence_human":
        text = (
            f"This content appears to be human-written. Our system analyzed the text "
            f"across multiple signals and found no significant indicators of AI authorship "
            f"(confidence: {pct}%)."
        )
    else:
        text = (
            f"The origin of this content is unclear. Our system detected mixed or "
            f"inconclusive signals and cannot make a confident determination "
            f"(confidence: {pct}%). This result should be interpreted with caution. "
            f"If you are the creator of this content and believe it is human-written, "
            f"you may submit an appeal using your submission ID."
        )

    if llm_failure:
        text += (
            " Note: the AI signal was temporarily unavailable; this result is based "
            "on surface-level analysis only."
        )

    return text
