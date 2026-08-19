"""Plotting helpers, extracted from the exploratory scripts."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import FIGURES_DIR  # noqa: E402


def plot_top_family_forecasts(df_store, final_forecasts, top_n=5, filename="scaled_forecasts.png"):
    """Grid of historical-vs-forecast plots for the top-N families by volume."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    top_families = df_store.groupby("family")["sales"].sum().nlargest(top_n).index

    plt.figure(figsize=(15, 10))
    for i, family in enumerate(top_families):
        plt.subplot(3, 2, i + 1)
        hist = df_store[df_store["family"] == family].tail(30)
        plt.plot(hist["date"], hist["sales"], label="Historical")
        fore = final_forecasts[final_forecasts["family"] == family]
        plt.plot(fore["date"], fore["prediction"], label="Forecast", marker="o")
        plt.title(f"Forecast: {family}")
        plt.xticks(rotation=45)
        plt.legend()
    plt.tight_layout()

    out = FIGURES_DIR / filename
    plt.savefig(out)
    plt.close()
    return out
