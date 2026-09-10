from pkg.discount import discount_rate


def test_non_member_has_no_discount():
    assert discount_rate(False, 200) == 0
