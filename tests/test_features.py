"""Tests for forecast.make_features: the features must not peek past the auction.

    python -m pytest
"""

import numpy as np
import pandas as pd
import pytest

import forecast

COLUMNS = ["price", "load", "fc_wind_onshore", "fc_wind_offshore", "fc_solar", "fc_total_generation"]


@pytest.fixture
def hourly():
    idx = pd.date_range("2025-03-01", periods=24 * 21, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(50, 10, (len(idx), len(COLUMNS))), index=idx, columns=COLUMNS)


def test_price_features_only_use_earlier_days(hourly):
    """Changing the price of one hour may only affect features of later days."""
    base = forecast.make_features(hourly)
    changed_at = hourly.index[24 * 10 + 15]          # some afternoon hour
    bumped = hourly.copy()
    bumped.loc[changed_at, "price"] += 1000
    after = forecast.make_features(bumped)
    diff = (base.fillna(-1) != after.fillna(-1)).any(axis=1)
    changed_day = changed_at.tz_convert(forecast.TZ).normalize()
    affected_days = after.index[diff].tz_convert(forecast.TZ).normalize()
    assert diff.any()
    assert (affected_days > changed_day).all(), "a feature used a price from the same day or later"


def test_load_features_lag_at_least_two_days(hourly):
    base = forecast.make_features(hourly)
    changed_at = hourly.index[24 * 10]
    bumped = hourly.copy()
    bumped.loc[changed_at, "load"] += 1000
    after = forecast.make_features(bumped)
    diff = (base.fillna(-1) != after.fillna(-1)).any(axis=1)
    assert (after.index[diff] >= changed_at + pd.Timedelta(hours=48)).all()


def test_forecast_columns_are_for_the_same_hour(hourly):
    feats = forecast.make_features(hourly)
    pd.testing.assert_series_equal(feats["fc_solar"], hourly["fc_solar"], check_names=False)


def test_calendar_uses_german_local_time_and_holidays():
    idx = pd.DatetimeIndex(["2025-12-24 23:00", "2025-07-01 10:00"], tz="UTC")
    df = pd.DataFrame(1.0, index=idx, columns=COLUMNS)
    feats = forecast.make_features(df)
    assert feats["hour"].tolist() == [0, 12]          # UTC+1 in winter, UTC+2 in summer
    assert feats["is_holiday"].tolist() == [1, 0]     # 25 Dec local time is a holiday


def test_previous_day_mean_is_the_whole_previous_local_day(hourly):
    feats = forecast.make_features(hourly)
    local = hourly.index.tz_convert(forecast.TZ)
    day = local.normalize()
    target = hourly.index[24 * 5 + 12]
    prev = day[24 * 5 + 12] - pd.Timedelta(days=1)
    expected = hourly["price"][day == prev].mean()
    assert feats.loc[target, "price_mean_prev_day"] == pytest.approx(expected)
