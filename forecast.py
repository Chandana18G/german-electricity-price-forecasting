"""
forecast.py - day-ahead price forecasting for Germany/Luxembourg, backtested honestly.

The EPEX SPOT day-ahead auction for day D closes at 12:00 on D-1, so every feature for an
hour of day D uses only information available at that moment:
    - prices of the same hour 1 and 7 days earlier (published before D-1's auction)
    - actual grid load 2 and 7 days earlier (actuals arrive with a delay)
    - the grid operators' day-ahead forecasts of wind, solar and total generation for D
    - calendar: hour, weekday, month, German public holiday

Backtest: rolling origin. For each of the last 12 complete months the models are trained
on everything before that month and predict every hour of it.

Usage:
    python smard.py       # download the data first
    python forecast.py    # -> results/metrics.json, results/predictions.csv.gz, figures/
"""

import json
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
TZ = "Europe/Berlin"
TEST_MONTHS = 12

NUMERIC = ["price_lag_24h", "price_lag_168h", "price_mean_prev_day", "load_lag_48h", "load_lag_168h",
           "fc_wind_onshore", "fc_wind_offshore", "fc_solar", "fc_total_generation",
           "fc_renewable_share"]
CALENDAR = ["hour", "weekday", "month", "is_holiday"]
FEATURES = NUMERIC + CALENDAR


def load_data(path=ROOT / "data" / "smard_hourly.csv.gz"):
    df = pd.read_csv(path, index_col="time_utc", parse_dates=True)
    full = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
    return df.reindex(full)   # a complete hourly grid, so shifts are exact hours


def make_features(df):
    """Feature table indexed like df. Each row only uses information available before the
    day-ahead auction for that hour's day (see the module docstring)."""
    local = df.index.tz_convert(TZ)
    out = pd.DataFrame(index=df.index)
    out["price_lag_24h"] = df["price"].shift(24)
    out["price_lag_168h"] = df["price"].shift(168)
    # Mean price of the previous local day: known in full before D-1's auction closes.
    day = local.normalize()
    daily_mean = df["price"].groupby(day).mean()
    out["price_mean_prev_day"] = (day - pd.Timedelta(days=1)).map(daily_mean).to_numpy()
    out["load_lag_48h"] = df["load"].shift(48)
    out["load_lag_168h"] = df["load"].shift(168)
    for col in ["fc_wind_onshore", "fc_wind_offshore", "fc_solar", "fc_total_generation"]:
        out[col] = df[col]   # forecasts for this hour, published before the auction
    renewable = df[["fc_wind_onshore", "fc_wind_offshore", "fc_solar"]].sum(axis=1, min_count=1)
    out["fc_renewable_share"] = renewable / df["fc_total_generation"]
    de_holidays = holidays.Germany(years=range(local.year.min(), local.year.max() + 1))
    out["hour"] = local.hour
    out["weekday"] = local.weekday
    out["month"] = local.month
    out["is_holiday"] = [int(d in de_holidays) for d in local.date]
    return out


def models():
    linear = make_pipeline(
        ColumnTransformer([("num", StandardScaler(), NUMERIC),
                           ("cal", OneHotEncoder(handle_unknown="ignore"), ["hour", "weekday", "month"]),
                           ("hol", "passthrough", ["is_holiday"])]),
        Ridge(alpha=1.0))
    boosting = HistGradientBoostingRegressor(
        max_iter=600, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=40,
        l2_regularization=1.0, categorical_features=[FEATURES.index(c) for c in ["hour", "weekday", "month"]],
        random_state=0)
    return {"Linear (ridge)": linear, "Gradient boosting": boosting}


def last_complete_month():
    return pd.Timestamp.now(tz=TZ).tz_localize(None).to_period("M") - 1


def backtest(df, feats, test_months=TEST_MONTHS):
    """Rolling-origin backtest over the last complete months. Returns one row per test hour
    with the actual price and every model's prediction."""
    data = feats.assign(price=df["price"]).dropna()
    local_month = data.index.tz_convert(TZ).tz_localize(None).to_period("M")
    months = pd.period_range(end=last_complete_month(), periods=test_months, freq="M")
    rows = []
    for month in months:
        train = data[local_month < month]
        test = data[local_month == month]
        pred = pd.DataFrame({"actual": test["price"],
                             "Naive: same hour yesterday": test["price_lag_24h"],
                             "Naive: same hour last week": test["price_lag_168h"]}, index=test.index)
        for name, model in models().items():
            model.fit(train[FEATURES], train["price"])
            pred[name] = model.predict(test[FEATURES])
        pred["month"] = str(month)
        rows.append(pred)
        print(f"  {month}: trained on {len(train):,} hours, tested on {len(test):,}", flush=True)
    return pd.concat(rows)


def summarize(pred):
    names = [c for c in pred.columns if c not in ("actual", "month")]
    err = {n: pred[n] - pred["actual"] for n in names}
    base = err["Naive: same hour yesterday"].abs().mean()
    overall = {n: {"mae": round(float(e.abs().mean()), 2), "rmse": round(float(np.sqrt((e ** 2).mean())), 2),
                   "skill_vs_yesterday": round(float(1 - e.abs().mean() / base), 3)} for n, e in err.items()}
    monthly = {n: pred.assign(ae=err[n].abs()).groupby("month")["ae"].mean().round(2).to_dict() for n in names}
    local_hour = pred.index.tz_convert(TZ).hour
    by_hour = {n: err[n].abs().groupby(local_hour).mean().round(2).to_dict() for n in names}
    neg = pred["actual"] < 0
    return {"overall": overall, "monthly_mae": monthly, "mae_by_hour": by_hour,
            "test_period": [str(pred.index.tz_convert(TZ).min()), str(pred.index.tz_convert(TZ).max())],
            "test_hours": int(len(pred)), "negative_price_hours": int(neg.sum()),
            "mae_on_negative_hours": {n: round(float(err[n][neg].abs().mean()), 2) for n in names}}


def importance(df, feats):
    """Permutation importance of the boosting model on the most recent test month."""
    data = feats.assign(price=df["price"]).dropna()
    month = data.index.tz_convert(TZ).tz_localize(None).to_period("M")
    last = last_complete_month()
    train, test = data[month < last], data[month == last]
    model = models()["Gradient boosting"].fit(train[FEATURES], train["price"])
    imp = permutation_importance(model, test[FEATURES], test["price"], scoring="neg_mean_absolute_error",
                                 n_repeats=5, random_state=0)
    return dict(sorted(zip(FEATURES, imp.importances_mean.round(2).tolist()), key=lambda kv: -kv[1]))


def main():
    df = load_data()
    feats = make_features(df)
    print(f"{df['price'].notna().sum():,} hours of prices, {df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d}")
    pred = backtest(df, feats)
    metrics = summarize(pred)
    metrics["permutation_importance_mae"] = importance(df, feats)
    metrics["data"] = {"hours": int(df["price"].notna().sum()),
                       "from": str(df.index.min().date()), "to": str(df.index.max().date())}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pred.to_csv(ROOT / "results" / "predictions.csv.gz", float_format="%.2f")
    for name, m in metrics["overall"].items():
        print(f"  {name:28s} MAE {m['mae']:6.2f}  RMSE {m['rmse']:6.2f}  skill {m['skill_vs_yesterday']:+.1%}")
    import figures
    figures.make_all(metrics, pred)


if __name__ == "__main__":
    main()
