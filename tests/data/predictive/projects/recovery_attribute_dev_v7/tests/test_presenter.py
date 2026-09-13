from pkg.presenter import title


def test_title(profile):
    assert title(profile) == "ADA"
