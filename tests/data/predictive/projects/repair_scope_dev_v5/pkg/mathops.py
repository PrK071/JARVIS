def ratio(total, count):
    return total / count


def mean(values):
    return ratio(sum(values), len(values))


def parse_count(raw):
    return raw


def process(raw):
    count = parse_count(raw)
    return ratio(10, count)


class Invoice:
    def __init__(self, tax=None):
        self.tax = tax

def invoice_total(invoice):
    return 10 + invoice.tax
