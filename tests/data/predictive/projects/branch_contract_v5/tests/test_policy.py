from pkg.policy import access_level, route


def test_member_access():
    assert access_level(False) == "member"


def test_disabled_route():
    assert route(False) == "off"
