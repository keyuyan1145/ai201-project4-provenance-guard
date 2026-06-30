import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import config
from audit import get_log, write_log_entry
from pipeline.classifier import classify
from pipeline.heuristic_signal import compute_heuristic_score
from pipeline.llm_signal import compute_llm_score

load_dotenv()

app = Flask(__name__)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)


@app.route("/submit", methods=["POST"])
@limiter.limit(config.RATE_LIMIT)
def submit():
    body = request.get_json(silent=True)

    if body is None:
        print("[WARN] POST /submit: request body is missing or not valid JSON")
        return jsonify({"error": "Request body must be valid JSON"}), 400

    content = body.get("content")
    creator_id = body.get("creator_id")

    if not isinstance(content, str) or not content.strip():
        print("[WARN] POST /submit: missing or empty 'content' field")
        return jsonify({"error": "Missing required field: content"}), 400

    if not isinstance(creator_id, str) or not creator_id.strip():
        print("[WARN] POST /submit: missing or empty 'creator_id' field")
        return jsonify({"error": "Missing required field: creator_id"}), 400

    label_id = str(uuid.uuid4())
    print(f"[INFO] POST /submit: submission received, creator_id={creator_id}, label_id={label_id}")

    # --- Signal 1: Statistical Heuristics ---
    heuristic_result = compute_heuristic_score(content)
    heuristic_score = heuristic_result["heuristic_score"]

    # --- Cost gate ---
    llm_attempted = heuristic_score >= config.HEURISTIC_GATE_THRESHOLD
    if not llm_attempted:
        print(
            f"[INFO] Gate: heuristic_score={heuristic_score:.3f} < "
            f"{config.HEURISTIC_GATE_THRESHOLD} — LLM call skipped"
        )
        llm_score = None
    else:
        # --- Signal 2: LLM Semantic Classifier ---
        llm_score = compute_llm_score(content)
        if llm_score is None:
            print("[INFO] LLM signal unavailable after all retries — single-signal fallback")

    # --- Classifier ---
    classifier_result = classify(heuristic_score, llm_score)
    weighted_score = classifier_result["weighted_score"]
    final_confidence_score = classifier_result["final_confidence_score"]
    llm_signal_available = classifier_result["llm_signal_available"]

    # --- Label assignment ---
    if not llm_attempted:
        # Gate closed: heuristic score below threshold is sufficient evidence of human authorship
        label = "high_confidence_human"
        print(f"[INFO] Gate closed (heuristic_score={heuristic_score:.3f}) — label forced to 'high_confidence_human'")
    elif llm_attempted and llm_score is None:
        # LLM was called but all retries failed — insufficient evidence for definitive label
        label = "uncertain"
        print("[INFO] LLM retries exhausted — label forced to 'uncertain'")
    elif weighted_score >= config.AI_SCORE_THRESHOLD and final_confidence_score >= config.CONFIDENCE_THRESHOLD:
        label = "high_confidence_ai"
    elif weighted_score <= config.HUMAN_SCORE_THRESHOLD and final_confidence_score >= config.CONFIDENCE_THRESHOLD:
        label = "high_confidence_human"
    else:
        label = "uncertain"

    print(
        f"[INFO] Label assigned: variant={label},"
        f" weighted_score={weighted_score:.3f},"
        f" final_confidence={final_confidence_score:.3f},"
        f" llm_signal_available={llm_signal_available}"
    )

    # --- Audit log ---
    now = datetime.now(timezone.utc)
    ms = now.microsecond // 1000
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ms:03d}Z"

    write_log_entry({
        "event_type": "classification",
        "content_id": label_id,
        "creator_id": creator_id,
        "timestamp": timestamp,
        "attribution": label,
        "confidence": final_confidence_score,
        "heuristic_score": heuristic_score,
        "llm_score": llm_score,
        "status": "classified",
    })
    print(f"[INFO] Submission {label_id} persisted to audit log")

    return jsonify({
        "label_id": label_id,
        "content_id": label_id,
        "weighted_score": weighted_score,
        "final_confidence_score": final_confidence_score,
        "attribution": final_confidence_score,
        "label": label,
        "llm_score": llm_score,
        "heuristic_score": heuristic_score,
    }), 200


@app.route("/log", methods=["GET"])
def log():
    limit = request.args.get("limit", default=20, type=int)
    event_type = request.args.get("event_type", default=None)
    entries = get_log(limit=limit, event_type=event_type)
    return jsonify({"entries": entries, "total": len(entries)}), 200


if __name__ == "__main__":
    app.run(debug=True)
