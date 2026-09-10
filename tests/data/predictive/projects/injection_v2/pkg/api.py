from .processor import normalize_record

def create_record(payload: dict) -> str:
    return normalize_record(payload)
