from pkg.loader import load_service


def test_loader():
    assert load_service().__class__.__name__ == "Service"
