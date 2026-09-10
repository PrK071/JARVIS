from pkg.worker import process

def test_processes_value():
    assert process(3) == "3"
