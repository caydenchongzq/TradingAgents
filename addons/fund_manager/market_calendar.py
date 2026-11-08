"""US equity session calendar (NYSE rules), New York time.

Rule-based so it needs no network or extra package. Covers the regular NYSE
holidays and the 13:00 early closes; one-off closures (national days of
mourning, weather) are not known in advance and must be passed by hand via
``--analysis-date`` if they ever happen.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo


def _ny_zone():
    """America/New_York when the IANA database exists, else None.

    python:3.13-slim has no tzdata, and subclassing datetime.tzinfo in that
    build raises TypeError under postponed annotations. The run script sets
    TZ=EST5EDT,M3.2.0,M11.1.0, so a naive local datetime is New York time.
    """
    try:
        zone = ZoneInfo("America/New_York")
        datetime.now(zone)
        return zone
    except Exception:
        return None


NY = _ny_zone()


def now_ny() -> datetime:
    return datetime.now(NY) if NY is not None else datetime.now()


def at_ny(d: date, t: time) -> datetime:
    return datetime.combine(d, t, NY) if NY is not None else datetime.combine(d, t)


def stamp_ny(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, NY) if NY is not None else datetime.fromtimestamp(ts)


def as_ny(dt: datetime) -> datetime:
    if NY is not None:
        return dt.astimezone(NY) if dt.tzinfo else dt.replace(tzinfo=NY)
    if dt.tzinfo:
        return dt.astimezone().replace(tzinfo=None)
    return dt
OPEN = time(9, 30)
CLOSE = time(16, 0)
EARLY_CLOSE = time(13, 0)
# Yahoo can show a closeless or partial daily bar for a while after the bell.
SETTLE_MINUTES = 30


def _easter(y: int) -> date:
    a = y % 19
    b, c = divmod(y, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(y, month, day + 1)


def _observed(d: date) -> date:
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def _nth_weekday(y: int, m: int, weekday: int, n: int) -> date:
    d = date(y, m, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _last_weekday(y: int, m: int, weekday: int) -> date:
    d = date(y, m + 1, 1) - timedelta(days=1) if m < 12 else date(y, 12, 31)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def nyse_holidays(y: int) -> set[date]:
    h: set[date] = set()
    new_year = date(y, 1, 1)
    # NYSE does not move a Saturday New Year's Day back to Dec 31.
    if new_year.weekday() == 6:
        h.add(new_year + timedelta(days=1))
    elif new_year.weekday() < 5:
        h.add(new_year)
    h.add(_nth_weekday(y, 1, 0, 3))          # Martin Luther King Jr. Day
    h.add(_nth_weekday(y, 2, 0, 3))          # Washington's Birthday
    h.add(_easter(y) - timedelta(days=2))    # Good Friday
    h.add(_last_weekday(y, 5, 0))            # Memorial Day
    if y >= 2022:
        h.add(_observed(date(y, 6, 19)))     # Juneteenth
    h.add(_observed(date(y, 7, 4)))          # Independence Day
    h.add(_nth_weekday(y, 9, 0, 1))          # Labor Day
    h.add(_nth_weekday(y, 11, 3, 4))         # Thanksgiving
    h.add(_observed(date(y, 12, 25)))        # Christmas
    return h


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in nyse_holidays(d.year)


def previous_trading_day(d: date) -> date:
    d -= timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def last_trading_day_on_or_before(d: date) -> date:
    return d if is_trading_day(d) else previous_trading_day(d)


def next_trading_day(d: date) -> date:
    d += timedelta(days=1)
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d


def close_time(d: date) -> time:
    thanksgiving = _nth_weekday(d.year, 11, 3, 4)
    early = {thanksgiving + timedelta(days=1), date(d.year, 12, 24), date(d.year, 7, 3)}
    return EARLY_CLOSE if d in early and is_trading_day(d) else CLOSE


def session_info(now: datetime | None = None) -> dict:
    """Where the US market is right now and which session bar is the latest final one.

    ``analysis_date`` is always the New York calendar date, so TradingAgents
    treats the run as live (its ``is_historical`` check compares with the
    container's ``date.today()``, which is NY time when TZ is set).
    """
    now = as_ny(now) if now else now_ny()
    today = now.date()
    if is_trading_day(today):
        open_dt = at_ny(today, OPEN)
        close_dt = at_ny(today, close_time(today))
        settled_dt = close_dt + timedelta(minutes=SETTLE_MINUTES)
        if now < open_dt:
            state, last_done = "pre_market", previous_trading_day(today)
        elif now < close_dt:
            state, last_done = "regular", previous_trading_day(today)
        elif now < settled_dt:
            state, last_done = "settling", previous_trading_day(today)
        else:
            state, last_done = "after_close", today
    else:
        state, last_done = "closed", previous_trading_day(today)
    return {
        "ny_now": now.isoformat(timespec="seconds"),
        "ny_date": today.isoformat(),
        "market_state": state,
        "intraday": state in ("regular", "settling"),
        "last_completed_session": last_done.isoformat(),
        "next_session": (today if state in ("pre_market", "regular", "settling") else next_trading_day(today)).isoformat(),
        "analysis_date": today.isoformat(),
    }


def expected_bar_for(analysis_date: date, now: datetime | None = None) -> date:
    """The newest session whose final bar should exist for ``analysis_date``."""
    info = session_info(now)
    if analysis_date.isoformat() == info["ny_date"]:
        return date.fromisoformat(info["last_completed_session"])
    return last_trading_day_on_or_before(analysis_date)
