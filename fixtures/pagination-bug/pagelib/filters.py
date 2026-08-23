"""Filtering helpers applied before pagination."""


def apply_filters(items, predicates=None):
    """Keep items satisfying every predicate."""
    if not predicates:
        return list(items)
    return [item for item in items if all(check(item) for check in predicates)]


def search(items, term, key=str):
    """Case-insensitive substring match over a projected field."""
    needle = (term or "").lower()
    if not needle:
        return list(items)
    return [item for item in items if needle in key(item).lower()]
