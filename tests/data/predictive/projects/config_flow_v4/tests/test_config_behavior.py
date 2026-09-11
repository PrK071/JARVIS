from app.client import backoff
from app.routing import route


def test_public_route():
    assert route(False) == "public"


def test_explicit_attempts():
    assert backoff(2) == [0, 1]
