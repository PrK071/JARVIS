from cycle2.alpha import alpha_value
from cycle3.red import red_value
from flow.operations import average, caller_supplies_text, consume_count


def test_average():
    assert average([2, 4]) == 3


def test_argument_source():
    assert caller_supplies_text() == 1


def test_return_contract():
    assert consume_count("2") == 5
