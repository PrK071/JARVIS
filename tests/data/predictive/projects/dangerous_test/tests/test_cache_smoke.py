from pkg.cache import cache_key


def test_cache_key_is_text():
    assert isinstance(cache_key(1, "us"), str)
