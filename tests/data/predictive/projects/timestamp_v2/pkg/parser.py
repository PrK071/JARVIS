from datetime import datetime

def parse_created(payload: dict) -> datetime:
    raw_created = payload["created_at"]
    return datetime.fromisoformat(raw_created)
