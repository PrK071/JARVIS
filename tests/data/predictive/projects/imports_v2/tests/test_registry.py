from pkg.registry import get_provider

def test_provider():
    assert get_provider() == "local"
