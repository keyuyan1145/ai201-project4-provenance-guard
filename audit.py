import json
import os
from datetime import datetime, timezone

import config


def _now_iso() -> str:
    now = datetime.now(timezone.utc)
    ms = now.microsecond // 1000
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ms:03d}Z"


def write_log_entry(entry: dict) -> None:
    dir_path = os.path.dirname(config.AUDIT_LOG_FILE)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    try:
        with open(config.AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        print(
            f"[INFO] Audit entry written:"
            f" event={entry.get('event_type')}, content_id={entry.get('content_id')}"
        )
    except Exception as exc:
        print(f"[ERROR] Failed to write to {config.AUDIT_LOG_FILE}: {exc}")


def get_log(limit: int = 20, event_type: str = None) -> list:
    if not os.path.exists(config.AUDIT_LOG_FILE):
        print(f"[WARN] GET /log: {config.AUDIT_LOG_FILE} does not exist — returning empty result")
        return []

    entries = []
    try:
        with open(config.AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if event_type is None or entry.get("event_type") == event_type:
                        entries.append(entry)
                except json.JSONDecodeError:
                    print(
                        f"[WARN] GET /log: skipping malformed line {line_num}"
                        f" in {config.AUDIT_LOG_FILE}"
                    )
    except Exception as exc:
        print(f"[ERROR] Failed to read {config.AUDIT_LOG_FILE}: {exc}")
        return []

    return entries[-limit:]
