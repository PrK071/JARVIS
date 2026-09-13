def divide_span(duration, slots):
    return duration / slots


def average_span(entries):
    return divide_span(sum(entries), len(entries))


def scheduled_span(duration, *, slots):
    return divide_span(duration, slots)
