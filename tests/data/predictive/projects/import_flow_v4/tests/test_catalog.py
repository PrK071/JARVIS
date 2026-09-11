def test_catalog_imports():
    from pkg.catalog import current_item

    assert callable(current_item)
