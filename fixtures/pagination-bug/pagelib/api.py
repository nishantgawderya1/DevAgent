"""Public pagination entry points."""

from pagelib.utils import clamp_page_size, offset_for, page_count


def paginate(items, page=1, page_size=None):
    """Return a single page of items.

    The requested page size is clamped before use so a caller cannot ask for an
    unbounded page.
    """
    size = clamp_page_size(page_size if page_size is not None else 20)
    start = offset_for(page, size)
    return list(items)[start : start + size]


def paginated_response(items, page=1, page_size=None):
    """Wrap a page of items with its navigation metadata."""
    size = clamp_page_size(page_size if page_size is not None else 20)
    return {
        "results": paginate(items, page=page, page_size=size),
        "page": page,
        "page_size": size,
        "total_pages": page_count(len(list(items)), size),
    }
