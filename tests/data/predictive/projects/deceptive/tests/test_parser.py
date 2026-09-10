from pkg.parser import parse_age


def test_parse_age():
    assert parse_age({"age": "42"}) == 42
