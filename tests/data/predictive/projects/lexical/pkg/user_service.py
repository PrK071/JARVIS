def normalize_user(user):
    email = user["email"]
    return {**user, "email": email.strip().lower()}
