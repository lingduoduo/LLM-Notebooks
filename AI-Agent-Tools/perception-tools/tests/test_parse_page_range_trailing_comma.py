"""Trailing commas in page_range must not crash parse_page_range."""

from perception_tools.document_processing_tools import parse_page_range


def test_trailing_comma_does_not_raise():
    assert parse_page_range("1,3,", 10) == [0, 2]


def test_duplicate_comma_does_not_raise():
    assert parse_page_range("1,,3", 10) == [0, 2]


def test_leading_comma_does_not_raise():
    assert parse_page_range(",1,5", 10) == [0, 4]


def test_normal_list_unchanged():
    assert parse_page_range("1,3,5", 10) == [0, 2, 4]


def test_range_with_trailing_comma():
    assert parse_page_range("1-3,", 10) == [0, 1, 2]
