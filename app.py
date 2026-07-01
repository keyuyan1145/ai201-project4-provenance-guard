import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import config
from audit import get_classification_entry, get_log, update_classification_entry, write_log_entry
from pipeline.classifier import classify
from pipeline.heuristic_signal import compute_heuristic_score
from pipeline.labels import generate_label_text
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

    content = body.get("text")
    creator_id = body.get("creator_id")

    if not isinstance(content, str) or not content.strip():
        print("[WARN] POST /submit: missing or empty 'text' field")
        return jsonify({"error": "Missing required field: text"}), 400

    if not isinstance(creator_id, str) or not creator_id.strip():
        print("[WARN] POST /submit: missing or empty 'creator_id' field")
        return jsonify({"error": "Missing required field: creator_id"}), 400

    label_id = str(uuid.uuid4())
    print(f"[INFO] POST /submit: submission received, creator_id={creator_id}, label_id={label_id}")

    # --- Signal 1: Statistical Heuristics ---
    heuristic_result = compute_heuristic_score(content)
    heuristic_score = heuristic_result["heuristic_score"]
    word_count = heuristic_result["word_count"]

    # --- Cost gate (two-sided) ---
    lower_gate_closed = heuristic_score < config.HEURISTIC_GATE_THRESHOLD
    upper_gate_closed = heuristic_score > config.HEURISTIC_AI_GATE_THRESHOLD

    if lower_gate_closed:
        print(
            f"[INFO] Lower gate: heuristic_score={heuristic_score:.3f} < "
            f"{config.HEURISTIC_GATE_THRESHOLD} — LLM skipped (confident human)"
        )
        llm_score = None
    elif upper_gate_closed:
        print(
            f"[INFO] Upper gate: heuristic_score={heuristic_score:.3f} > "
            f"{config.HEURISTIC_AI_GATE_THRESHOLD} — LLM skipped (confident AI)"
        )
        llm_score = None
    else:
        # --- Signal 2: LLM Semantic Classifier ---
        llm_score = compute_llm_score(content)
        if llm_score is None:
            print("[INFO] LLM signal unavailable after all retries — single-signal fallback")

    # --- Classifier ---
    classifier_result = classify(heuristic_score, llm_score, word_count)
    weighted_score = classifier_result["weighted_score"]
    signal_agreement = classifier_result["signal_agreement"]
    llm_signal_available = classifier_result["llm_signal_available"]

    # --- Label assignment ---
    if lower_gate_closed:
        label = "high_confidence_human"
        print(f"[INFO] Lower gate closed (heuristic_score={heuristic_score:.3f}) — label forced to 'high_confidence_human'")
    elif upper_gate_closed:
        label = "high_confidence_ai"
        print(f"[INFO] Upper gate closed (heuristic_score={heuristic_score:.3f}) — label forced to 'high_confidence_ai'")
    elif llm_score is None:
        # LLM was called but all retries failed — insufficient evidence for definitive label
        label = "uncertain"
        print("[INFO] LLM retries exhausted — label forced to 'uncertain'")
    elif weighted_score >= config.AI_SCORE_THRESHOLD:
        label = "high_confidence_ai"
    elif weighted_score <= config.HUMAN_SCORE_THRESHOLD:
        label = "high_confidence_human"
    else:
        label = "uncertain"

    llm_failure = not lower_gate_closed and not upper_gate_closed and llm_score is None
    label_text = generate_label_text(label, weighted_score, llm_failure=llm_failure)

    print(
        f"[INFO] Label assigned: variant={label},"
        f" weighted_score={weighted_score:.3f},"
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
        "text": content,
        "attribution": label,
        "confidence": weighted_score,
        "heuristic_score": heuristic_score,
        "llm_score": llm_score,
        "agreement_score": signal_agreement,
        "status": "classified",
    })
    print(f"[INFO] Submission {label_id} persisted to audit log")

    return jsonify({
        "label_id": label_id,
        "content_id": label_id,
        "text": content,
        "weighted_score": weighted_score,
        "attribution": weighted_score,
        "label": label,
        "label_text": label_text,
        "llm_score": llm_score,
        "heuristic_score": heuristic_score,
        "agreement_score": signal_agreement,
    }), 200


@app.route("/appeal", methods=["POST"])
def appeals():
    body = request.get_json(silent=True)

    if body is None:
        print("[WARN] POST /appeal: request body is missing or not valid JSON")
        return jsonify({"error": "Request body must be valid JSON"}), 400

    content_id = body.get("content_id")
    creator_reasoning = body.get("creator_reasoning")

    if not isinstance(content_id, str) or not content_id.strip():
        print("[WARN] POST /appeal: missing or empty 'content_id' field")
        return jsonify({"error": "Missing required field: content_id"}), 400

    if not isinstance(creator_reasoning, str) or not creator_reasoning.strip():
        print("[WARN] POST /appeal: missing or empty 'creator_reasoning' field")
        return jsonify({"error": "Missing required field: creator_reasoning"}), 400

    existing = get_classification_entry(content_id)
    if existing is None:
        print(f"[WARN] POST /appeal: no classification entry found for content_id={content_id}")
        return jsonify({"error": f"No submission found with content_id: {content_id}"}), 404

    if existing.get("status") == "under_review":
        print(f"[WARN] POST /appeal: appeal already submitted for content_id={content_id}")
        return jsonify({"error": "An appeal for this submission has already been submitted"}), 409

    appeal_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    ms = now.microsecond // 1000
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ms:03d}Z"

    updated = update_classification_entry(content_id, {
        "status": "under_review",
        "appeal_reasoning": creator_reasoning,
    })

    if not updated:
        print(f"[ERROR] POST /appeal: failed to update audit entry for content_id={content_id}")
        return jsonify({"error": "Failed to update submission status"}), 500

    write_log_entry({
        "event_type": "appeal",
        "appeal_id": appeal_id,
        "content_id": content_id,
        "creator_reasoning": creator_reasoning,
        "timestamp": timestamp,
        "status": "under_review",
    })

    print(f"[INFO] Appeal {appeal_id} received for content_id={content_id}")

    return jsonify({
        "appeal_id": appeal_id,
        "content_id": content_id,
        "status": "under_review",
        "message": "Your appeal has been received and is under review.",
        "timestamp": timestamp,
    }), 200


@app.route("/log", methods=["GET"])
def log():
    limit = request.args.get("limit", default=20, type=int)
    event_type = request.args.get("event_type", default=None)
    entries = get_log(limit=limit, event_type=event_type)
    return jsonify({"entries": entries, "total": len(entries)}), 200


if __name__ == "__main__":
    app.run(debug=True)
