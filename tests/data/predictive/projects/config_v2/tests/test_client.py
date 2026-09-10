from pkg.client import retry_delays

def test_three_retry_delays(monkeypatch):
    monkeypatch.setattr("pkg.client.RETRY_LIMIT", 3)
    assert retry_delays() == [0, 1, 2]
