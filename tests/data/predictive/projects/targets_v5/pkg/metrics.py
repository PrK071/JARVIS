def ratio(total, count):
    return total / count


def average(values):
    return ratio(sum(values), len(values))


def parse_limit(raw):
    return raw


def bounded(raw):
    limit = parse_limit(raw)
    return ratio(20, limit)


class Cart:
    def __init__(self, fee=None):
        self.fee = fee

def cart_total(cart):
    return 20 + cart.fee
