from pagelib.api import paginate, paginated_response


def test_paginate_returns_the_requested_page():
    items = list(range(100))

    assert paginate(items, page=1, page_size=10) == list(range(10))
    assert paginate(items, page=2, page_size=10) == list(range(10, 20))


def test_paginate_never_exceeds_the_maximum_page_size():
    items = list(range(500))

    page = paginate(items, page=1, page_size=1000)

    assert len(page) == 50


def test_paginated_response_reports_its_page_size():
    items = list(range(500))

    response = paginated_response(items, page=1, page_size=1000)

    assert response["page_size"] == 50
    assert len(response["results"]) == 50


def test_paginate_handles_an_empty_collection():
    assert paginate([], page=1, page_size=10) == []
