import uuid

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import config

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

    # Hardcoded response — real pipeline wired in M4
    return jsonify({
        "label_id": label_id,
        "weighted_score": 0.76,
        "final_confidence_score": 0.82,
        "label": "high_confidence_ai",
        "llm_score": 0.78,
        "heuristic_score": 0.72,
    }), 200


if __name__ == "__main__":
    app.run(debug=True)
