from pkg.worker import process


def test_process():
    assert process("ok") == "ok"
