from pkg.user_service import normalize_user


def test_normalizes_email():
    assert normalize_user({"email": " A@B.COM "})["email"] == "a@b.com"
