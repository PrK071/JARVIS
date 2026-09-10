from pkg.processor import normalize_record

def test_normalizes_name():
    assert normalize_record({"name": " Ada "}) == "Ada"
