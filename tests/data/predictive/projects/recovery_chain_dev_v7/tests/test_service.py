from pkg.service import render_account


def test_missing_account():
    assert render_account("missing") == "UNKNOWN"
