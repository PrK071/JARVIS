from app.handler import handle


def test_delay():
    assert handle(10) == 15
