# pagelib

A small pagination library.

```python
from pagelib.api import paginate, paginated_response

paginate(range(500), page=1, page_size=10)
paginated_response(range(500), page=2, page_size=25)
```

## Running the tests

```bash
pip install -r requirements.txt
pytest -q
```
