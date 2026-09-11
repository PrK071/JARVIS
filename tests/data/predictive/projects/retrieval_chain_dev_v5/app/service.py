from .repository import fetch


def render(record_id):
    record = fetch(record_id)
    return record["name"].upper()
