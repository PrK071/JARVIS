from pkg.handler import handle_event

def test_handler_is_callable():
    assert callable(handle_event)
