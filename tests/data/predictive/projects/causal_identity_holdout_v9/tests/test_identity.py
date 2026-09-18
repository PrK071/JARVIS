from mesh.north import northern
from pipeline.math import calculate_units, density, submit_bad_payload
from ring.east import eastern


def test_density():
    assert density([3, 5]) == 4


def test_argument_source():
    assert submit_bad_payload() == 2


def test_return_contract():
    assert calculate_units("3") == 4
