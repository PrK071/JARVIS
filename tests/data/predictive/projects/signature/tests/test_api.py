from pkg.api import send_message


def test_send_message():
    assert send_message("a@b.test", "Hello", "Body")["subject"] == "Hello"
