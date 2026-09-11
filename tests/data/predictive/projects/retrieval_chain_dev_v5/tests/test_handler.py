from app.handler import handle


def test_absent_record():
    assert handle("absent") == "UNKNOWN"
