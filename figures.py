"""
figures.py - charts from results/metrics.json and results/predictions.csv.gz, in figures/.

Usage:
    python figures.py     # redraw from the saved results
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
TZ = "Europe/Berlin"

# Dark theme matching the portfolio; colours checked for contrast and colour-blind separation.
BG, GRID, AXIS, TEXT, MUTED = "#110e1a", "#2a2240", "#4a4066", "#ece8f5", "#a59dbb"
COLOURS = {"Naive: same hour yesterday": "#8b5cf6", "Naive: same hour last week": "#c2850a",
           "Linear (ridge)": "#d946ef", "Gradient boosting": "#0f9f8f"}


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "axes.edgecolor": AXIS, "axes.labelcolor": MUTED, "axes.titlecolor": TEXT,
        "axes.titlesize": 17, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 34, "axes.labelsize": 13, "axes.grid": True, "grid.color": GRID,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 12, "ytick.labelsize": 12,
        "xtick.major.size": 0, "ytick.major.size": 0, "text.color": TEXT,
        "legend.frameon": False, "legend.fontsize": 12, "font.size": 12.5,
    })


def subtitle(ax, text):
    ax.text(0, 1.025, text, transform=ax.transAxes, color=MUTED, fontsize=12.5, va="bottom")


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=150)
    plt.close(fig)
    print(f"Wrote figures/{name}.png")


def accuracy_chart(m):
    names = list(COLOURS)
    mae = [m["overall"][n]["mae"] for n in names]
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    bars = ax.barh(range(len(names)), mae, color=[COLOURS[n] for n in names], height=0.6)
    for i, (b, n) in enumerate(zip(bars, names)):
        skill = m["overall"][n]["skill_vs_yesterday"]
        label = f"{mae[i]:.1f} EUR/MWh" + ("" if n == names[0] else f"   ({skill:+.0%} vs yesterday)")
        ax.text(b.get_width() + max(mae) * 0.01, i, label, va="center", fontsize=13, color=TEXT)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlim(0, max(mae) * 1.45)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean absolute error (EUR/MWh) - lower is better")
    ax.set_title("Forecasting tomorrow's hourly price")
    start, end = (t[:10] for t in m["test_period"])
    subtitle(ax, f"Backtest over {m['test_hours']:,} hours ({start} to {end}); each month predicted by a model "
                 "trained only on earlier months")
    fig.tight_layout()
    save(fig, "accuracy")


def example_week(pred):
    local = pred.tz_convert(TZ)
    weekly_spread = local["actual"].resample("W-SUN").agg(lambda s: s.max() - s.min())
    week_end = weekly_spread.iloc[-8:].idxmax()   # the most volatile of the last 8 weeks
    week = local[(local.index > week_end - pd.Timedelta(days=7)) & (local.index <= week_end)]
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.plot(week.index, week["actual"], color=TEXT, lw=2.4, label="Actual price")
    ax.plot(week.index, week["Gradient boosting"], color=COLOURS["Gradient boosting"], lw=2.2, label="Gradient boosting forecast")
    ax.plot(week.index, week["Naive: same hour yesterday"], color=COLOURS["Naive: same hour yesterday"], lw=1.6,
            ls="--", label="Same hour yesterday")
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_ylabel("EUR/MWh")
    ax.legend(loc="upper left", ncol=3)
    ax.set_title("One volatile week, hour by hour")
    mae_b = (week["Gradient boosting"] - week["actual"]).abs().mean()
    mae_n = (week["Naive: same hour yesterday"] - week["actual"]).abs().mean()
    subtitle(ax, f"{week.index.min():%d %b} to {week.index.max():%d %b %Y} (German time) · "
                 f"MAE this week: boosting {mae_b:.1f}, yesterday's price {mae_n:.1f} EUR/MWh")
    ax.set_ylim(min(week[["actual", "Gradient boosting"]].min().min(), 0) - 15,
                week[["actual", "Gradient boosting", "Naive: same hour yesterday"]].max().max() * 1.22)
    fig.autofmt_xdate()
    fig.tight_layout()
    save(fig, "example_week")


def hourly_error_chart(m):
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    for n in ["Naive: same hour yesterday", "Gradient boosting"]:
        by_hour = {int(h): v for h, v in m["mae_by_hour"][n].items()}
        hours = sorted(by_hour)
        ax.plot(hours, [by_hour[h] for h in hours], color=COLOURS[n], lw=3, marker="o", ms=5, label=n)
    ax.set_xticks(range(0, 24, 2), [f"{h:02d}:00" for h in range(0, 24, 2)])
    ax.set_xlabel("Hour of the day (German time)")
    ax.set_ylabel("Mean absolute error (EUR/MWh)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    ax.set_title("Where the error is: the morning ramp and the evening peak")
    subtitle(ax, "Error by delivery hour over the backtest year")
    fig.tight_layout()
    save(fig, "error_by_hour")


LABELS = {"price_lag_24h": "Price, same hour yesterday", "price_lag_168h": "Price, same hour last week",
          "price_mean_prev_day": "Yesterday's average price", "load_lag_48h": "Load, 2 days earlier",
          "load_lag_168h": "Load, 7 days earlier", "fc_wind_onshore": "Forecast: onshore wind",
          "fc_wind_offshore": "Forecast: offshore wind", "fc_solar": "Forecast: solar",
          "fc_total_generation": "Forecast: total generation", "fc_renewable_share": "Forecast: renewable share",
          "hour": "Hour of day", "weekday": "Weekday", "month": "Month", "is_holiday": "Public holiday"}


def importance_chart(m):
    imp = m["permutation_importance_mae"]
    names = list(imp)[:10][::-1]
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.barh([LABELS[n] for n in names], [imp[n] for n in names], color=COLOURS["Gradient boosting"], height=0.6)
    for i, n in enumerate(names):
        ax.text(max(imp[n], 0) + max(imp.values()) * 0.01, i, f"{imp[n]:+.1f}", va="center", fontsize=12, color=TEXT)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Increase in MAE (EUR/MWh) when the feature is shuffled")
    ax.set_title("What the model relies on")
    subtitle(ax, "Permutation importance of the gradient-boosting model on the most recent test month")
    fig.tight_layout()
    save(fig, "importance")


def make_all(metrics=None, pred=None):
    if metrics is None:
        metrics = json.loads((ROOT / "results" / "metrics.json").read_text())
    if pred is None:
        pred = pd.read_csv(ROOT / "results" / "predictions.csv.gz", index_col=0, parse_dates=True)
    apply_style()
    accuracy_chart(metrics)
    example_week(pred)
    hourly_error_chart(metrics)
    importance_chart(metrics)


if __name__ == "__main__":
    make_all()
