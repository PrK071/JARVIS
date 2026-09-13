from inventory.provider import acquire_record, acquire_quantity


def forward_record():
    return acquire_record()


def transform_quantity():
    return acquire_quantity() + 1
