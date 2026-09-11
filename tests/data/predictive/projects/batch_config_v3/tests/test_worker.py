from app.mode import retries
from app.worker import batches


def test_defaults():
    assert batches(2) == [0, 1]
    assert retries(False) == 0
