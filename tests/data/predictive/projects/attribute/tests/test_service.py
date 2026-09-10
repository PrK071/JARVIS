from pkg.service import display_name


def test_display_name(profile_user):
    assert display_name(profile_user) == "ADA"
