from pkg.mathops import Invoice, invoice_total, mean, process


def test_mean_empty():
    assert mean([]) == 0


def test_process_count():
    assert process(0) == 10


def test_invoice_total():
    assert invoice_total(Invoice()) == 10
