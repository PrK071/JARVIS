def find_user(rows, user_id):
    for row in rows:
        if row["id"] == user_id:
            return row["name"]
    return None
