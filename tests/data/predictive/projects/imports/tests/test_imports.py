def test_accounts_imports():
    from pkg.accounts import current_account

    assert callable(current_account)
