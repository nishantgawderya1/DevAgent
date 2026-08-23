**Title:** `paginate()` returns one item more than the maximum page size

---

### What happens

Asking for a page larger than the maximum returns 51 items instead of 50.

```python
>>> from pagelib.api import paginate
>>> len(paginate(range(500), page=1, page_size=1000))
51
```

`paginated_response` reports the same wrong size in its metadata:

```python
>>> from pagelib.api import paginated_response
>>> paginated_response(range(500), page=1, page_size=1000)["page_size"]
51
```

### What should happen

`MAX_PAGE_SIZE` is 50, so an over-large request should be clamped to exactly 50
items — and the reported `page_size` should agree.

### Reproducing

```bash
pip install -r requirements.txt
pytest -q
```

Two tests fail:

```
FAILED tests/test_api.py::test_paginate_never_exceeds_the_maximum_page_size
FAILED tests/test_api.py::test_paginated_response_reports_its_page_size
```

Both report `assert 51 == 50`.

### Notes

Pages under the maximum are fine, so the clamping boundary looks like the place
to check rather than the slicing itself.
