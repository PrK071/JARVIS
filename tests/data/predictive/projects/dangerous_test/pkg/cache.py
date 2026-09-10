def cache_key(user_id, region):
    if not region:
        raise ValueError("region is required")
    return f"{region}:{user_id}"
