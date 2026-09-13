from billing.handler import checkout


def test_void_invoice():
    assert checkout("void") == 0
