from app.storage import fetch_customer
from app.presenter import label


def customer_label(customer_id):
    customer = fetch_customer(customer_id)
    return label(customer)
