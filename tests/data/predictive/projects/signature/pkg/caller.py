from .api import send_message


def notify_admin(body):
    return send_message("admin@example.invalid", body)
