from app.settings import BATCH_SIZE


def batches(size=BATCH_SIZE):
    return list(range(size))
