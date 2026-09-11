from pkg.adapter import adapt


def test_adapter_identity():
    assert adapt("ok") == "ok"
