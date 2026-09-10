from pkg import settings

def test_retry_setting_exists():
    assert hasattr(settings, "RETRY_LIMIT")
