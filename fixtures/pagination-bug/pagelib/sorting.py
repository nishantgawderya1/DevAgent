"""Ordering helpers applied before pagination."""


def sort_items(items, key=None, descending=False):
    """Stable sort with an optional projection."""
    return sorted(items, key=key, reverse=descending) if key else sorted(items, reverse=descending)


def stable_rank(items, key):
    """Map each item to its 1-based rank under a projection."""
    ordered = sort_items(items, key=key)
    return {item: index for index, item in enumerate(ordered, start=1)}
