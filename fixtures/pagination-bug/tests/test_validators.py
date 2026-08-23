import pytest

from pagelib.validators import ValidationError, validate_page, validate_page_size


def test_validate_page_rejects_non_positive():
    assert validate_page(3) == 3
    with pytest.raises(ValidationError):
        validate_page(0)


def test_validate_page_size_allows_none():
    assert validate_page_size(None) is None
    assert validate_page_size(25) == 25
    with pytest.raises(ValidationError):
        validate_page_size(-1)
