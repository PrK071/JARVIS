from media.player import seek


def test_seek():
    assert seek({"duration": "10"}) == 15
