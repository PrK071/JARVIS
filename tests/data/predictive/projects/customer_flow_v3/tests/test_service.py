from app.service import customer_label


def test_customer_label():
    assert customer_label("known") == "Ada"
