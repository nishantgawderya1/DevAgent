from pagelib.filters import apply_filters, search


def test_apply_filters_requires_every_predicate():
    items = [1, 2, 3, 4, 5, 6]

    assert apply_filters(items, [lambda n: n % 2 == 0]) == [2, 4, 6]
    assert apply_filters(items, [lambda n: n % 2 == 0, lambda n: n > 3]) == [4, 6]


def test_search_is_case_insensitive():
    assert search(["Alpha", "beta", "GAMMA"], "mm") == ["GAMMA"]
    assert search(["Alpha", "beta", "GAMMA"], "A") == ["Alpha", "beta", "GAMMA"]
    assert search(["Alpha"], "") == ["Alpha"]
