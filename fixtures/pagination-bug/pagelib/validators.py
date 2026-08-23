"""Request validation for the pagination endpoints."""


class ValidationError(ValueError):
    """Raised when pagination parameters do not make sense."""


def validate_page(page):
    """Pages are 1-based."""
    if not isinstance(page, int) or page < 1:
        raise ValidationError(f"page must be a positive integer, got {page!r}")
    return page


def validate_page_size(page_size):
    """Page size must be a positive integer when supplied."""
    if page_size is None:
        return None
    if not isinstance(page_size, int) or page_size < 1:
        raise ValidationError(f"page_size must be a positive integer, got {page_size!r}")
    return page_size
