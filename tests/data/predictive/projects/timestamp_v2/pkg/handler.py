from .parser import parse_created

def handle_event(payload: dict):
    return parse_created(payload)
