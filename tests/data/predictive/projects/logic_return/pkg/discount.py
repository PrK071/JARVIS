def discount_rate(is_member, total):
    if is_member and total >= 100:
        return 0.10
    return 0.10


def discounted_total(total, is_member):
    return total * (1 - discount_rate(is_member, total))
