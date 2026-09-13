from inventory.presenter import quantity_total, record_name
from scheduler.window import average_span, scheduled_span
from cache.pool import next_capacity


def test_record_name():
    assert record_name() == "ready"


def test_quantity_total():
    assert quantity_total() == 15


def test_average_span():
    assert average_span([]) == 0


def test_scheduled_span():
    assert scheduled_span(8, slots=0) == 8


def test_capacity():
    assert next_capacity() == 65
