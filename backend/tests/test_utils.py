import os
import sys

import pytest


BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from utils import clean_and_normalize_urls


@pytest.mark.parametrize(
    "raw,expected",
    [
        (
            ["https://x.com/someuser/status/1234567890"],
            ["https://x.com/someuser/status/1234567890"],
        ),
        (
            ["https://x.com/i/status/987654321"],
            ["https://x.com/i/status/987654321"],
        ),
        (
            ["x.com/handle/status/13579"],
            ["https://x.com/handle/status/13579"],
        ),
    ],
)
def test_clean_and_normalize_urls_basic_cases(raw, expected):
    assert clean_and_normalize_urls(raw) == expected


def test_clean_and_normalize_urls_mixed_text():
    raw = [
        "Check this https://x.com/alpha/status/1 and also x.com/bravo/status/2 thanks",
    ]
    expected = [
        "https://x.com/alpha/status/1",
        "https://x.com/bravo/status/2",
    ]
    assert clean_and_normalize_urls(raw) == expected


def test_clean_and_normalize_urls_dedupes():
    raw = [
        "https://x.com/alpha/status/1",
        "Some extra x.com/alpha/status/1 text",
        "x.com/alpha/status/1",
    ]
    expected = ["https://x.com/alpha/status/1"]
    assert clean_and_normalize_urls(raw) == expected
