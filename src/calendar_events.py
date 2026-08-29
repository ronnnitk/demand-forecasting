"""Festival, holiday and season features — the demand drivers the calendar
integers alone cannot express.

The dataset is Ecuadorian retail, and its largest demand swings are not weekly
seasonality (already captured by Fourier terms) but *events*: the Christmas
build-up, New Year, Carnival, Mother's Day, national holidays and the pre-holiday
stock-up that precedes each. A raw ``month`` integer cannot say "three days
before Christmas"; a ``days_to_holiday`` feature can, and that proximity is where
the grocery uplift actually lives.

Everything here is derived purely from the date, so — like the calendar
features — it is fully known in advance and safe at any forecast horizon. The
holiday set is generated in code (fixed national days plus Easter-derived movable
feasts via the Anonymous Gregorian computus), so the module needs no external
holidays file and works on a fresh checkout.

Scope note: this encodes Ecuador's *national* civic and commercial calendar. It
deliberately omits regional/local holidays (which vary by city and would need the
per-store locale) and treats the April 2016 earthquake as a one-off documented
shock rather than a recurring seasonal driver.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

from .config import DATE_COL

# The 2016 Ecuador earthquake and its ~6-week relief-demand window.
EARTHQUAKE_START = pd.Timestamp("2016-04-16")
EARTHQUAKE_END = pd.Timestamp("2016-05-31")

# --- Fixed-date national holidays (month, day) -> name -------------------------
FIXED_HOLIDAYS: dict[tuple[int, int], str] = {
    (1, 1): "new_year",
    (5, 1): "labour_day",
    (5, 24): "battle_of_pichincha",
    (8, 10): "independence_quito",
    (10, 9): "independence_guayaquil",
    (11, 2): "day_of_the_dead",
    (11, 3): "independence_cuenca",
    (12, 25): "christmas",
}

# Stable integer codes for the holiday categorical (0 is reserved for "no holiday").
HOLIDAY_NAMES = [
    "new_year", "carnival", "good_friday", "labour_day", "battle_of_pichincha",
    "independence_quito", "independence_guayaquil", "day_of_the_dead",
    "independence_cuenca", "christmas",
]
HOLIDAY_CODES = {name: i + 1 for i, name in enumerate(HOLIDAY_NAMES)}

# Retail "festival seasons": windows that move demand beyond the single holiday.
# Stable codes for the festival categorical (0 = ordinary day).
FESTIVAL_NAMES = [
    "christmas_season", "new_year_season", "carnival", "valentines",
    "mothers_day", "black_friday", "back_to_school",
]
FESTIVAL_CODES = {name: i + 1 for i, name in enumerate(FESTIVAL_NAMES)}


def easter_sunday(year: int) -> date:
    """Gregorian Easter via the Anonymous (Meeus/Jones/Butcher) algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 22 * m) // 451
    month = (h + m - 7 * n + 114) // 31
    day = ((h + m - 7 * n + 114) % 31) + 1
    return date(year, month, day)


@lru_cache(maxsize=32)
def holidays_for_year(year: int) -> dict[date, str]:
    """Every national holiday in a year: fixed days plus Easter-derived feasts."""
    out = {date(year, m, d): name for (m, d), name in FIXED_HOLIDAYS.items()}
    easter = easter_sunday(year)
    out[easter - timedelta(days=48)] = "carnival"       # Carnival Monday
    out[easter - timedelta(days=47)] = "carnival"       # Carnival Tuesday
    out[easter - timedelta(days=2)] = "good_friday"
    return out


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The n-th ``weekday`` (Mon=0) of a month, e.g. 2nd Sunday of May."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


@lru_cache(maxsize=32)
def festival_windows_for_year(year: int) -> list[tuple[date, date, str]]:
    """(start, end, name) windows for the commercial festival seasons."""
    mothers = _nth_weekday(year, 5, 6, 2)               # 2nd Sunday of May
    black_friday = _nth_weekday(year, 11, 4, 4)         # 4th Friday of November
    easter = easter_sunday(year)
    return [
        (date(year, 12, 1), date(year, 12, 24), "christmas_season"),
        (date(year, 12, 26), date(year, 12, 31), "new_year_season"),
        (easter - timedelta(days=52), easter - timedelta(days=46), "carnival"),
        (date(year, 2, 10), date(year, 2, 14), "valentines"),
        (mothers - timedelta(days=6), mothers, "mothers_day"),
        (black_friday, black_friday + timedelta(days=3), "black_friday"),
        # Coastal school year starts in April, highland in September; cover both.
        (date(year, 4, 1), date(year, 4, 15), "back_to_school"),
        (date(year, 9, 1), date(year, 9, 15), "back_to_school"),
    ]


def build_holiday_frame(start, end) -> pd.DataFrame:
    """A per-date table of holiday and festival attributes over ``[start, end]``.

    Columns: ``date``, ``holiday_name``, ``festival_name``. Absent days are
    simply not holidays/festivals (filled downstream).
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    years = range(start.year, end.year + 1)

    holidays: dict[date, str] = {}
    festivals: dict[date, str] = {}
    for year in years:
        holidays.update(holidays_for_year(year))
        for w_start, w_end, name in festival_windows_for_year(year):
            for offset in range((w_end - w_start).days + 1):
                festivals[w_start + timedelta(days=offset)] = name

    calendar = pd.DataFrame({DATE_COL: pd.date_range(start, end, freq="D")})
    day = calendar[DATE_COL].dt.date
    calendar["holiday_name"] = day.map(holidays)
    calendar["festival_name"] = day.map(festivals)
    return calendar


def _proximity(is_event: np.ndarray, horizon: int = 14) -> tuple[np.ndarray, np.ndarray]:
    """Signed distances (in days) to the nearest event before and after each day.

    Returns ``(days_to_next, days_since_last)``, both clipped to ``horizon`` so a
    day far from any event doesn't emit a huge number the model reads as trend.
    """
    n = len(is_event)
    idx = np.flatnonzero(is_event)
    days_to = np.full(n, horizon, dtype="int16")
    days_since = np.full(n, horizon, dtype="int16")
    if idx.size == 0:
        return days_to, days_since

    positions = np.arange(n)
    nxt = np.searchsorted(idx, positions, side="left")
    has_next = nxt < idx.size
    days_to[has_next] = np.clip(idx[nxt[has_next]] - positions[has_next], 0, horizon)

    prv = np.searchsorted(idx, positions, side="right") - 1
    has_prev = prv >= 0
    days_since[has_prev] = np.clip(positions[has_prev] - idx[prv[has_prev]], 0, horizon)
    return days_to, days_since


def event_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append holiday / festival / season features to a frame with a ``date``.

    Designed to be called from :func:`src.features.calendar_features`. Operates
    on the distinct dates in the frame, so it costs the same for one series or
    the whole panel.
    """
    out = df.copy()
    dates = out[DATE_COL]

    calendar = build_holiday_frame(dates.min(), dates.max())
    calendar["is_holiday"] = calendar["holiday_name"].notna().astype("int8")
    calendar["holiday_code"] = (
        calendar["holiday_name"].map(HOLIDAY_CODES).fillna(0).astype("int16")
    )
    calendar["is_festival"] = calendar["festival_name"].notna().astype("int8")
    calendar["festival_code"] = (
        calendar["festival_name"].map(FESTIVAL_CODES).fillna(0).astype("int16")
    )

    # Holiday proximity: the stock-up before and the lull after.
    to_next, since_last = _proximity(calendar["is_holiday"].to_numpy(bool))
    calendar["days_to_holiday"] = to_next
    calendar["days_since_holiday"] = since_last
    calendar["is_holiday_eve"] = (to_next == 1).astype("int8")

    # Named festival flags the model (and the recommender) can key on directly.
    for name in ("christmas_season", "mothers_day", "black_friday"):
        calendar[f"is_{name}"] = (calendar["festival_name"] == name).astype("int8")

    # Ecuadorian wet season (Jan-May) tracks coastal demand patterns; a coarse
    # but real seasonal split for an equatorial market with no summer/winter.
    calendar["wet_season"] = calendar[DATE_COL].dt.month.isin(range(1, 6)).astype("int8")

    # The 16 April 2016 (Mw 7.8) earthquake drove a documented, weeks-long spike
    # in grocery/water/first-aid demand as relief efforts ran. Flagging the
    # response window lets the model attribute that spike to the shock rather
    # than mis-learning it as seasonality it will wrongly repeat every April.
    quake = (calendar[DATE_COL] >= EARTHQUAKE_START) & (calendar[DATE_COL] <= EARTHQUAKE_END)
    calendar["is_earthquake_response"] = quake.astype("int8")

    feature_cols = [
        "is_holiday", "holiday_code", "days_to_holiday", "days_since_holiday",
        "is_holiday_eve", "is_festival", "festival_code",
        "is_christmas_season", "is_mothers_day", "is_black_friday", "wet_season",
        "is_earthquake_response",
    ]
    merged = out.merge(calendar[[DATE_COL, *feature_cols]], on=DATE_COL, how="left")
    merged.index = out.index
    return merged


# Feature columns this module contributes, so features.py can register the
# categoricals without hard-coding the list in two places.
EVENT_FEATURE_COLUMNS = [
    "is_holiday", "holiday_code", "days_to_holiday", "days_since_holiday",
    "is_holiday_eve", "is_festival", "festival_code",
    "is_christmas_season", "is_mothers_day", "is_black_friday", "wet_season",
    "is_earthquake_response",
]
EVENT_CATEGORICAL_FEATURES = ["holiday_code", "festival_code"]
