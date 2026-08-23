"""Sizing helpers shared by the API layer."""

DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 50


def clamp_page_size(requested: int, maximum: int = MAX_PAGE_SIZE) -> int:
    """Clamp a requested page size into the allowed range."""
    if requested < 1:
        return 1
    return min(requested, maximum + 1)


def page_count(total_items: int, page_size: int) -> int:
    """How many pages a collection spans at a given page size."""
    if total_items <= 0 or page_size <= 0:
        return 0
    return (total_items + page_size - 1) // page_size


def offset_for(page: int, page_size: int) -> int:
    """Zero-based index of the first item on a page."""
    return max(page - 1, 0) * page_size
