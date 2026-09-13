def ratio(total, count):
    return total / count


def mean(values):
    return ratio(sum(values), len(values))


def render_ratio(total, *, count):
    return ratio(total, count)
