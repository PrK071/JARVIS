def display_name(user):
    return user.profile.name.upper()


def welcome(user):
    return f"Welcome {display_name(user)}"
