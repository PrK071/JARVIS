def find_account(rows, account_id):
    for row in rows:
        if row["id"] == account_id:
            return row["label"]
    return None
