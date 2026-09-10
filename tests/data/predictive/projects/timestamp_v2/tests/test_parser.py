from pkg.parser import parse_created

def test_parses_iso_timestamp():
    assert parse_created({"created_at": "2026-01-02"}).year == 2026
