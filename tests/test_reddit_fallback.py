"""Reddit's retired RSS source remains explicitly unavailable without I/O."""

from unittest.mock import patch

import pytest

from tradingagents.dataflows.vendors import reddit


@pytest.mark.unit
def test_reddit_reports_retired_source_without_network_requests():
    with patch.object(reddit, "urlopen", create=True) as urlopen:
        result = reddit.fetch_reddit_posts("NVDA")

    assert "unavailable" in result.lower()
    assert "rss was retired" in result.lower()
    assert "not an absence of discussion" in result.lower()
    urlopen.assert_not_called()


@pytest.mark.unit
def test_reddit_preserves_crypto_community_context():
    assert reddit.subreddits_for("BTC-USD") == (
        "Bitcoin", "CryptoCurrency", "CryptoMarkets",
    )
    result = reddit.fetch_reddit_posts("BTC-USD")
    assert "r/Bitcoin" in result
    assert "r/CryptoCurrency" in result


@pytest.mark.unit
def test_reddit_fetch_signature_keeps_historical_and_screening_callers():
    result = reddit.fetch_reddit_posts(
        "NVDA", ("stocks",), start_date="2026-05-01", end_date="2026-05-08",
        screen=lambda posts: ([], "screened"),
    )
    assert "unavailable" in result.lower()
    assert "not an absence of discussion" in result.lower()
