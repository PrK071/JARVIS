from .storage import read_row


def fetch(record_id):
    return read_row(record_id)
