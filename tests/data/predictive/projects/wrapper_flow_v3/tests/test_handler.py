from app.handler import handle


def test_handle():
    assert handle({"age": 2}) == 3
