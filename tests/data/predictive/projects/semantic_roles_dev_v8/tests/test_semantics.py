from commerce.view import invoice_amount, invoice_label
from metrics.core import mean, render_ratio
from settings.client import timeout_seconds


def test_invoice_label():
    assert invoice_label() == "paid"


def test_invoice_amount():
    assert invoice_amount() == 24


def test_mean():
    assert mean([]) == 0


def test_keyword_count():
    assert render_ratio(10, count=0) == 10


def test_timeout():
    assert timeout_seconds() == 31
