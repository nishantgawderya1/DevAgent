"""Turn pages into plain dictionaries for transport."""


def serialize_item(item):
    """Best-effort conversion of one item."""
    if hasattr(item, "as_dict"):
        return item.as_dict()
    if isinstance(item, dict):
        return dict(item)
    return {"value": item}


def serialize_page(page):
    """Serialize every item in a page."""
    return [serialize_item(item) for item in page]
