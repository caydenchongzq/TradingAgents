"""Reddit sentiment source status.

Reddit retired public RSS feeds in November 2026. Its replacement API access
terms do not establish that this external research use is eligible, so the
fetcher reports the source as unavailable until a supported access path is
confirmed. The string return contract is retained for the sentiment analyst.
"""

from __future__ import annotations

from collections.abc import Iterable

from tradingagents.dataflows.symbols import crypto_base

# Retained for callers that describe the communities relevant to a symbol.
DEFAULT_SUBREDDITS = ("wallstreetbets", "stocks", "investing")
CRYPTO_SUBREDDITS = ("CryptoCurrency", "CryptoMarkets")
_COIN_SUBREDDITS = {"BTC": "Bitcoin", "ETH": "ethereum", "SOL": "solana"}


def subreddits_for(ticker: str) -> tuple[str, ...]:
    """Return the communities historically used for this ticker's context."""
    base = crypto_base(ticker)
    if not base:
        return DEFAULT_SUBREDDITS
    coin = _COIN_SUBREDDITS.get(base)
    return ((coin,) if coin else ()) + CRYPTO_SUBREDDITS


def fetch_reddit_posts(
    ticker: str,
    subreddits: Iterable[str] | None = None,
    *,
    limit_per_sub: int = 5,
    timeout: float = 10.0,
    start_date: str | None = None,
    end_date: str | None = None,
    screen=None,
) -> str:
    """Return an explicit unavailable marker without contacting retired RSS."""
    del limit_per_sub, timeout, start_date, end_date, screen
    communities = tuple(subreddits) if subreddits is not None else subreddits_for(ticker)
    label = ", ".join(f"r/{name}" for name in communities)
    return (
        f"<Reddit unavailable: public RSS was retired; no supported access path "
        f"is configured for {ticker.upper()} across {label}. This is not an "
        f"absence of discussion>"
    )
