from inventory.tunnel import forward_record, transform_quantity


def record_name():
    return forward_record().name


def quantity_total():
    return transform_quantity() * 3
