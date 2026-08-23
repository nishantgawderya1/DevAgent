from pagelib.utils import offset_for, page_count


def test_page_count_rounds_up():
    assert page_count(100, 10) == 10
    assert page_count(101, 10) == 11
    assert page_count(0, 10) == 0


def test_offset_for_is_zero_based():
    assert offset_for(1, 20) == 0
    assert offset_for(3, 20) == 40
