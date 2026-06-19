"""
plots_annual.py
===============
Plotting module for full-year (or multi-month) rolling-horizon results.

All plots aggregate to daily or monthly resolution — individual 15-min
timestep plots are unreadable at 35,000+ points.

Call from main.py:
    from plots_annual import plot_annual_results
    plot_annual_results(results_all, econ_results, P_max, wind_capacity_kw,
                        price_H2=price_H2[0], save_dir="figures/")
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import os
from typing import List, Dict, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def plot_annual_results(
    results_all:      List[Dict],
    econ_results:     Dict,
    P_max:            float,
    wind_capacity_kw: float,
    price_H2:         float   = 4.5,
    save_dir:         str     = "figures/",
    show:             bool    = True,
):
    """
    Generate all annual analysis plots.

    Parameters
    ----------
    results_all      : per-day results list from main.py
    econ_results     : dict from EconomicsAnalysis.to_dict()
    P_max            : rated electrolyzer power [kW]
    wind_capacity_kw : net wind farm capacity [kW]
    price_H2         : baseline H2 price [EUR/kg]  (for annotations)
    save_dir         : directory to save figures (created if missing)
    show             : call plt.show() after each figure
    """
    os.makedirs(save_dir, exist_ok=True)

    df = _build_daily_df(results_all, P_max, wind_capacity_kw)

    _plot_power_overview(df,      save_dir, show)
    _plot_revenue_breakdown(df,   save_dir, show, price_H2)
    _plot_electrolyzer_ops(df,    save_dir, show, P_max)
    _plot_hydrogen(df,            save_dir, show)
    _plot_market_prices(df,       save_dir, show)
    _plot_duration_curves(results_all, P_max, wind_capacity_kw, save_dir, show)
    _plot_monthly_summary(df,     save_dir, show)
    _plot_economics_summary(econ_results, save_dir, show)




# ──────────────────────────────────────────────────────────────────────────────
# Build daily aggregated DataFrame
# ──────────────────────────────────────────────────────────────────────────────

def _build_daily_df(results_all, P_max, wind_capacity_kw):
    dt = 0.25   # h per slot

    rows = []
    for d in results_all:
        T       = len(d["P_DA"])
        on_mask = np.array(d["z_on_RT"]) > 0.5

        # Energy [MWh/day]
        wind_mwh    = np.sum(d["wind_real"])  * dt / 1000
        elec_mwh    = np.sum(d["P_elec_RT"]) * dt / 1000
        export_mwh  = np.sum(d["P_grid"])     * dt / 1000
        da_bid_mwh  = np.sum(d["P_DA"])       * dt / 1000
        curt_mwh    = np.sum(d["P_curt_RT"])  * dt / 1000
        buy_mwh     = np.sum(d["P_buy_RT"])   * dt / 1000

        # Hydrogen [kg/day]
        h2_kg       = np.sum(d["h_sell_RT"]) * dt

        # FCR [kW — daily mean of block values]
        da_fcr_mean = np.mean(d["DA_FCR"])

        # Revenue [EUR/day]
        price_da    = np.asarray(d["price_DA"])
        price_imb   = np.asarray(d["price_imb"])
        price_h2    = np.asarray(d["price_H2"])
        price_fcr   = np.asarray(d["price_FCR"])
        DA_FCR      = np.asarray(d["DA_FCR"])
        imbalance   = np.asarray(d["imbalance"])

        rev_da      = float(np.sum(np.array(d["P_DA"])      * price_da  * dt / 1000))
        rev_imb     = float(np.sum(imbalance                * price_imb * dt / 1000))
        rev_h2      = float(np.sum(np.array(d["h_sell_RT"]) * price_h2  * dt))
        cost_buy    = float(np.sum(np.array(d["P_buy_RT"])  * price_da  * dt / 1000))
        rev_fcr     = float(np.sum(DA_FCR * price_fcr * 4.0))   # 4 h per block
        net_rev     = rev_da + rev_imb + rev_h2 - cost_buy + rev_fcr

        # Operational
        util_pct    = 100.0 * on_mask.mean()
        curt_pct    = 100.0 * np.sum(d["P_curt_RT"]) / max(np.sum(d["wind_real"]), 1)
        imb_mwh     = float(np.sum(imbalance) * dt / 1000)
        mean_price  = float(np.mean(price_da))
        mean_imb_p  = float(np.mean(price_imb))

        # Min load on timesteps when on [% of P_max]
        p_elec = np.array(d["P_elec_RT"])
        min_load_pct = 100 * p_elec[on_mask].min() / P_max if on_mask.any() else np.nan

        rows.append({
            "day":          d.get("day", len(rows)),
            "wind_mwh":     wind_mwh,
            "elec_mwh":     elec_mwh,
            "export_mwh":   export_mwh,
            "da_bid_mwh":   da_bid_mwh,
            "curt_mwh":     curt_mwh,
            "buy_mwh":      buy_mwh,
            "h2_kg":        h2_kg,
            "da_fcr_kw":    da_fcr_mean,
            "rev_da":       rev_da,
            "rev_imb":      rev_imb,
            "rev_h2":       rev_h2,
            "cost_buy":     cost_buy,
            "rev_fcr":      rev_fcr,
            "net_rev":      net_rev,
            "util_pct":     util_pct,
            "curt_pct":     curt_pct,
            "imb_mwh":      imb_mwh,
            "mean_price_da": mean_price,
            "mean_price_imb": mean_imb_p,
            "min_load_pct": min_load_pct,
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.date_range("2025-03-05", periods=len(df), freq="D")

    #df["date"] = pd.to_datetime([d["date"] for d in results_all])
    df["month"] = df["date"].dt.month
    df["week"]  = df["date"].dt.isocalendar().week.astype(int)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Plot 1: Power overview — daily wind, electrolyzer, export, curtailment
# ──────────────────────────────────────────────────────────────────────────────

def _plot_power_overview(df, save_dir, show):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Panel 1: Wind and electrolyzer
    axes[0].fill_between(df["date"], df["wind_mwh"] / 24,
                         alpha=0.3, color="steelblue", label="Wind (avg MW)")
    axes[0].plot(df["date"], df["elec_mwh"] / 24,
                 color="darkorange", lw=1.2, label="Electrolyzer (avg MW)")
    axes[0].set_ylabel("Average power [MW]")
    axes[0].set_title("Daily average wind and electrolyzer power")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    # Panel 2: Grid export vs DA bid
    axes[1].plot(df["date"], df["da_bid_mwh"] / 24,
                 color="navy", lw=1.0, alpha=0.8, label="DA bid (avg MW)")
    axes[1].plot(df["date"], df["export_mwh"] / 24,
                 color="royalblue", lw=1.0, alpha=0.8, label="RT export (avg MW)")
    axes[1].fill_between(df["date"],
                         df["da_bid_mwh"] / 24, df["export_mwh"] / 24,
                         alpha=0.2, color="red", label="Imbalance")
    axes[1].set_ylabel("Average power [MW]")
    axes[1].set_title("DA bid vs RT export")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    # Panel 3: Curtailment
    axes[2].bar(df["date"], df["curt_mwh"],
                color="crimson", alpha=0.7, width=0.8, label="Curtailed [MWh/day]")
    axes[2].set_ylabel("Curtailment [MWh/day]")
    axes[2].set_title("Daily wind curtailment")
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=0.3, axis="y")
    _fmt_date_axis(axes[2])

    fig.suptitle("Power flows overview", fontsize=13, fontweight="medium")
    plt.tight_layout()
    _save(fig, save_dir, "01_power_overview.png", show)


# ──────────────────────────────────────────────────────────────────────────────
# Plot 2: Revenue breakdown — stacked daily bars
# ──────────────────────────────────────────────────────────────────────────────

def _plot_revenue_breakdown(df, save_dir, show, price_H2):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    # Stacked revenue bars
    rev_cols   = ["rev_da", "rev_fcr", "rev_h2", "rev_imb"]
    rev_labels = ["DA export", "FCR", "H₂ sales", "Imbalance"]
    rev_colors = ["#1565C0", "#2E7D32", "#F57F17", "#6A1B9A"]

    bottom = np.zeros(len(df))
    for col, label, color in zip(rev_cols, rev_labels, rev_colors):
        vals = df[col].values
        pos  = np.maximum(vals, 0)
        neg  = np.minimum(vals, 0)
        axes[0].bar(df["date"], pos, bottom=bottom,
                    color=color, alpha=0.85, width=0.8, label=label)
        bottom += pos

    # Grid cost as negative
    axes[0].bar(df["date"], -df["cost_buy"].values,
                color="#B71C1C", alpha=0.7, width=0.8, label="Grid cost")

    # Net revenue line
    ax0r = axes[0].twinx()
    cumrev = df["net_rev"].cumsum() / 1e6
    ax0r.plot(df["date"], cumrev, color="black", lw=1.5,
              ls="--", label="Cumulative net (MEUR)")
    ax0r.set_ylabel("Cumulative net revenue [MEUR]")
    ax0r.legend(loc="upper left", fontsize=8)

    axes[0].axhline(0, color="black", lw=0.8, alpha=0.4)
    axes[0].set_ylabel("Daily revenue [EUR]")
    axes[0].set_title("Daily revenue breakdown")
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)
    axes[0].grid(alpha=0.3, axis="y")

    # Revenue share (rolling 7-day)
    total_pos = df[rev_cols].clip(lower=0).sum(axis=1).replace(0, np.nan)
    for col, label, color in zip(rev_cols, rev_labels, rev_colors):
        share = df[col].clip(lower=0).rolling(7, min_periods=1).mean() \
                / total_pos.rolling(7, min_periods=1).mean() * 100
        axes[1].plot(df["date"], share, label=label, color=color, lw=1.2)

    axes[1].set_ylabel("Revenue share [%]  (7-day rolling)")
    axes[1].set_title("Revenue mix over time")
    axes[1].legend(loc="upper right", ncol=2, fontsize=8)
    axes[1].set_ylim(0, 100)
    axes[1].grid(alpha=0.3)
    _fmt_date_axis(axes[1])

    fig.suptitle("Revenue breakdown", fontsize=13, fontweight="medium")
    plt.tight_layout()
    _save(fig, save_dir, "02_revenue_breakdown.png", show)


# ──────────────────────────────────────────────────────────────────────────────
# Plot 3: Electrolyzer operations
# ──────────────────────────────────────────────────────────────────────────────

def _plot_electrolyzer_ops(df, save_dir, show, P_max):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Utilisation
    axes[0].plot(df["date"], df["util_pct"],
                 color="darkorange", lw=1.2)
    axes[0].fill_between(df["date"], df["util_pct"],
                         alpha=0.25, color="darkorange")
    axes[0].axhline(df["util_pct"].mean(), color="red", ls="--", lw=1,
                    label=f"Mean: {df['util_pct'].mean():.1f}%")
    axes[0].set_ylabel("Utilisation [%]")
    axes[0].set_title("Electrolyzer daily utilisation (% of timesteps on)")
    axes[0].set_ylim(0, 105)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Average load when on (as % of P_max)
    avg_load = df["elec_mwh"] * 1000 / (24 * P_max) * 100   # cf × 100
    axes[1].plot(df["date"], avg_load, color="teal", lw=1.2)
    axes[1].axhline(avg_load.mean(), color="teal", ls="--", lw=1,
                    label=f"Mean: {avg_load.mean():.1f}%")
    axes[1].set_ylabel("Daily capacity factor [%]")
    axes[1].set_title("Electrolyzer daily capacity factor")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # FCR capacity
    axes[2].bar(df["date"], df["da_fcr_kw"] / 1000,
                color="#2E7D32", alpha=0.7, width=0.8)
    axes[2].set_ylabel("Mean FCR capacity [MW]")
    axes[2].set_title("Daily mean FCR capacity committed")
    axes[2].grid(alpha=0.3, axis="y")
    _fmt_date_axis(axes[2])

    fig.suptitle("Electrolyzer operations", fontsize=13, fontweight="medium")
    plt.tight_layout()
    _save(fig, save_dir, "03_electrolyzer_ops.png", show)


# ──────────────────────────────────────────────────────────────────────────────
# Plot 4: Hydrogen production
# ──────────────────────────────────────────────────────────────────────────────

def _plot_hydrogen(df, save_dir, show):
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    axes[0].bar(df["date"], df["h2_kg"],
                color="#F57F17", alpha=0.8, width=0.8, label="Daily H₂ [kg]")
    roll7 = df["h2_kg"].rolling(7, min_periods=1).mean()
    axes[0].plot(df["date"], roll7, color="darkred", lw=1.5,
                 label="7-day rolling mean")
    axes[0].set_ylabel("H₂ sold [kg/day]")
    axes[0].set_title("Daily hydrogen production and sales")
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis="y")

    cum_h2 = df["h2_kg"].cumsum() / 1000
    axes[1].plot(df["date"], cum_h2, color="#E65100", lw=2)
    axes[1].set_ylabel("Cumulative H₂ [t]")
    axes[1].set_title("Cumulative hydrogen production")
    axes[1].grid(alpha=0.3)
    _fmt_date_axis(axes[1])

    fig.suptitle("Hydrogen production", fontsize=13, fontweight="medium")
    plt.tight_layout()
    _save(fig, save_dir, "04_hydrogen.png", show)


# ──────────────────────────────────────────────────────────────────────────────
# Plot 5: Market prices
# ──────────────────────────────────────────────────────────────────────────────

def _plot_market_prices(df, save_dir, show):
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    axes[0].plot(df["date"], df["mean_price_da"],
                 color="steelblue", lw=1.0, alpha=0.7, label="Daily mean DA price")
    axes[0].plot(df["date"],
                 df["mean_price_da"].rolling(30, min_periods=1).mean(),
                 color="navy", lw=2, label="30-day rolling mean")
    axes[0].axhline(0, color="black", lw=0.8, alpha=0.3)
    axes[0].set_ylabel("DA price [EUR/MWh]")
    axes[0].set_title("Day-ahead electricity price")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(df["date"], df["mean_price_imb"],
                 color="darkorchid", lw=1.0, alpha=0.7, label="Daily mean imbalance price")
    axes[1].plot(df["date"],
                 df["mean_price_imb"].rolling(30, min_periods=1).mean(),
                 color="indigo", lw=2, label="30-day rolling mean")
    axes[1].axhline(0, color="black", lw=0.8, alpha=0.3)
    axes[1].set_ylabel("Imbalance price [EUR/MWh]")
    axes[1].set_title("Imbalance settlement price")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    _fmt_date_axis(axes[1])

    fig.suptitle("Market prices", fontsize=13, fontweight="medium")
    plt.tight_layout()
    _save(fig, save_dir, "05_market_prices.png", show)


# ──────────────────────────────────────────────────────────────────────────────
# Plot 6: Duration curves — key operational insight
# ──────────────────────────────────────────────────────────────────────────────

def _plot_duration_curves(results_all, P_max, wind_capacity_kw, save_dir, show):
    """
    Duration curves show how often each power/price level is reached.
    Much more informative than time series for annual analysis.
    """
    dt = 0.25
    # Concatenate all timesteps
    wind_all  = np.concatenate([d["wind_real"]  for d in results_all]) / 1000  # MW
    elec_all  = np.concatenate([d["P_elec_RT"]  for d in results_all]) / 1000
    export_all= np.concatenate([d["P_grid"]     for d in results_all]) / 1000
    price_all = np.concatenate([d["price_DA"]   for d in results_all])
    curt_all  = np.concatenate([d["P_curt_RT"]  for d in results_all]) / 1000

    def duration(arr):
        return np.sort(arr)[::-1]

    pct = np.linspace(0, 100, len(wind_all))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Wind duration curve
    axes[0].plot(pct, duration(wind_all), color="steelblue", lw=2, label="Wind")
    axes[0].axhline(wind_capacity_kw / 1000, color="steelblue", ls=":",
                    lw=1, label=f"Rated ({wind_capacity_kw/1000:.0f} MW)")
    axes[0].set_xlabel("% of time exceeded")
    axes[0].set_ylabel("Power [MW]")
    axes[0].set_title("Wind power duration curve")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Electrolyzer duration curve
    on_mask  = elec_all > 0.01 * P_max / 1000
    axes[1].plot(pct, duration(elec_all), color="darkorange", lw=2)
    axes[1].axhline(P_max / 1000, color="darkorange", ls=":",
                    lw=1, label=f"Rated ({P_max/1000:.0f} MW)")
    axes[1].axhline(P_max * 0.20 / 1000, color="red", ls="--",
                    lw=1, label=f"P_min 20% ({P_max*0.20/1000:.0f} MW)")
    frac_on = 100 * on_mask.mean()
    axes[1].axvline(100 - frac_on, color="gray", ls="--", lw=1,
                    label=f"Off {100-frac_on:.0f}% of time")
    axes[1].set_xlabel("% of time exceeded")
    axes[1].set_ylabel("Power [MW]")
    axes[1].set_title("Electrolyzer load duration curve")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    # DA price duration curve
    pos_pct = 100 * (price_all >= 0).mean()
    axes[2].plot(pct, duration(price_all), color="navy", lw=2)
    axes[2].axhline(0, color="black", lw=0.8, alpha=0.4)
    axes[2].axvline(pos_pct, color="red", ls="--", lw=1,
                    label=f"Negative price: {100-pos_pct:.1f}% of time")
    axes[2].set_xlabel("% of time exceeded")
    axes[2].set_ylabel("Price [EUR/MWh]")
    axes[2].set_title("DA price duration curve")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.suptitle("Load and price duration curves", fontsize=13, fontweight="medium")
    plt.tight_layout()
    _save(fig, save_dir, "06_duration_curves.png", show)


# ──────────────────────────────────────────────────────────────────────────────
# Plot 7: Monthly summary — bar chart aggregation
# ──────────────────────────────────────────────────────────────────────────────

def _plot_monthly_summary(df, save_dir, show):
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    mo = df.groupby("month").agg(
        wind_gwh   =("wind_mwh",  "sum"),
        elec_gwh   =("elec_mwh",  "sum"),
        h2_t       =("h2_kg",     lambda x: x.sum() / 1000),
        curt_gwh   =("curt_mwh",  "sum"),
        util_pct   =("util_pct",  "mean"),
        net_rev_meur=("net_rev",  lambda x: x.sum() / 1e6),
        rev_da_meur =("rev_da",   lambda x: x.sum() / 1e6),
        rev_h2_meur =("rev_h2",   lambda x: x.sum() / 1e6),
        rev_fcr_meur=("rev_fcr",  lambda x: x.sum() / 1e6),
    ).reset_index()

    mo["wind_gwh"]  /= 1000   # MWh → GWh
    mo["elec_gwh"]  /= 1000
    mo["curt_gwh"]  /= 1000

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    x      = mo["month"].values - 1   # 0-indexed for tick labels
    labels = [month_names[i] for i in x]

    def barchart(ax, vals, color, ylabel, title, ylim=None):
        ax.bar(x, vals, color=color, alpha=0.8, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
        if ylim:
            ax.set_ylim(ylim)

    barchart(axes[0], mo["wind_gwh"],    "steelblue", "GWh",    "Monthly wind generation")
    barchart(axes[1], mo["elec_gwh"],    "darkorange","GWh",    "Monthly electrolyzer consumption")
    barchart(axes[2], mo["h2_t"],        "#F57F17",   "t H₂",  "Monthly H₂ production")
    barchart(axes[3], mo["curt_gwh"],    "crimson",   "GWh",    "Monthly curtailment")
    barchart(axes[4], mo["util_pct"],    "teal",      "%",      "Mean daily utilisation", (0, 100))

    # Stacked revenue
    bottom = np.zeros(len(mo))
    for col, label, color in [
        ("rev_da_meur",  "DA export", "#1565C0"),
        ("rev_h2_meur",  "H₂ sales",  "#F57F17"),
        ("rev_fcr_meur", "FCR",       "#2E7D32"),
    ]:
        axes[5].bar(x, mo[col].values, bottom=bottom,
                    color=color, alpha=0.85, width=0.6, label=label)
        bottom += mo[col].values
    axes[5].plot(x, mo["net_rev_meur"].values, "ko--", lw=1.5,
                 markersize=4, label="Net revenue")
    axes[5].set_xticks(x)
    axes[5].set_xticklabels(labels, fontsize=8)
    axes[5].set_ylabel("MEUR")
    axes[5].set_title("Monthly revenue")
    axes[5].legend(fontsize=7)
    axes[5].grid(alpha=0.3, axis="y")

    fig.suptitle("Monthly summary", fontsize=13, fontweight="medium")
    plt.tight_layout()
    _save(fig, save_dir, "07_monthly_summary.png", show)


# ──────────────────────────────────────────────────────────────────────────────
# Plot 8: Economics summary — single KPI figure
# ──────────────────────────────────────────────────────────────────────────────

def _plot_economics_summary(econ_results, save_dir, show):
    r = econ_results
    if not r:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Revenue pie
    rev_labels = ["DA export", "FCR", "H₂ sales", "Imbalance"]
    rev_keys   = ["da_export_revenue_yr_eur", "fcr_revenue_yr_eur",
                  "h2_revenue_yr_eur",        "imbalance_revenue_yr_eur"]
    rev_colors = ["#1565C0", "#2E7D32", "#F57F17", "#6A1B9A"]
    rev_vals   = [max(r.get(k, 0), 0) for k in rev_keys]

    if sum(rev_vals) > 0:
        wedges, texts, autotexts = axes[0].pie(
            rev_vals, labels=rev_labels, colors=rev_colors,
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1},
        )
        for at in autotexts:
            at.set_fontsize(8)
    axes[0].set_title("Revenue mix (annualised)")

    # Cost waterfall
    items  = ["CAPEX\n(annualised)", "OPEX", "Grid\ncost", "EBITDA\n(net)"]
    vals   = [
        -r.get("annualised_capex_eur", 0) / 1e6,
        -r.get("total_opex_yr_eur",    0) / 1e6,
        -r.get("grid_cost_yr_eur",     0) / 1e6,
         r.get("total_revenue_yr_eur", 0) / 1e6
         - r.get("total_opex_yr_eur", 0) / 1e6,
    ]
    colors = ["#B71C1C" if v < 0 else "#1B5E20" for v in vals]
    axes[1].bar(items, vals, color=colors, alpha=0.8, edgecolor="white")
    axes[1].axhline(0, color="black", lw=0.8, alpha=0.4)
    axes[1].set_ylabel("MEUR/yr")
    axes[1].set_title("Annual cost–revenue waterfall")
    axes[1].grid(alpha=0.3, axis="y")

    # KPI text box
    irr_str  = f"{r['irr']*100:.1f}%"  if r.get("irr") is not None else "N/A"
    pb_str   = f"{r.get('payback_yr', float('inf')):.1f} yr" \
               if np.isfinite(r.get("payback_yr", float("inf"))) else "∞"
    lcoh_str = f"{r.get('lcoh_eur_per_kg', 0):.2f} EUR/kg"
    npv_str  = f"{r.get('npv_eur', 0)/1e6:.1f} MEUR"
    flh_str  = f"{r.get('flh_electrolyzer_hr', 0):.0f} h/yr"
    cf_str   = f"{r.get('CF_electrolyzer', 0)*100:.1f}%" \
               if "CF_electrolyzer" in r else "—"

    kpi_text = (
        f"NPV        {npv_str}\n"
        f"IRR        {irr_str}\n"
        f"Payback    {pb_str}\n"
        f"LCOH       {lcoh_str}\n"
        f"FLH        {flh_str}\n"
        f"El. CF     {cf_str}\n"
        f"CAPEX      {r.get('total_capex_eur',0)/1e6:.0f} MEUR"
    )
    axes[2].axis("off")
    axes[2].text(0.1, 0.5, kpi_text,
                 transform=axes[2].transAxes,
                 fontsize=12, fontfamily="monospace",
                 verticalalignment="center",
                 bbox=dict(boxstyle="round,pad=0.6",
                           facecolor="#F5F5F5", edgecolor="#BDBDBD"))
    axes[2].set_title("Key performance indicators")

    fig.suptitle("Economic summary", fontsize=13, fontweight="medium")
    plt.tight_layout()
    _save(fig, save_dir, "08_economics_summary.png", show)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _fmt_date_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")


def _save(fig, save_dir, filename, show):
    path = os.path.join(save_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Standalone H2 price sensitivity plot (for Step 2 in the workflow)
# ──────────────────────────────────────────────────────────────────────────────

def override_h2_price(results_all: List[Dict], price: float) -> List[Dict]:
    """
    Return a copy of results_all with price_H2 set to a fixed value.
    Does NOT re-run the operational model — dispatch is unchanged.
    Used for post-hoc sensitivity analysis on H2 price.
    """
    out = []
    for day in results_all:
        d = dict(day)
        d["price_H2"] = np.full(len(d["price_H2"]), price)
        out.append(d)
    return out


def plot_h2_sensitivity(
    results_all:  List[Dict],
    econ_cfg_base,
    h2_prices:    List[float],
    P_el_kw:      float,
    save_dir:     str  = "figures/",
    show:         bool = True,
):
    """
    Run post-hoc H2 price sensitivity and plot NPV, LCOH, and revenue mix.
    No new solver calls — uses cached operational results.
    """
    from Economics import EconomicsAnalysis
    import copy

    os.makedirs(save_dir, exist_ok=True)
    rows = []
    for price in h2_prices:
        repriced = override_h2_price(results_all, price)
        cfg      = copy.deepcopy(econ_cfg_base)
        cfg.electrolyzer_capacity_kw = P_el_kw
        kpis     = EconomicsAnalysis(cfg).compute(repriced)
        rows.append({
            "h2_price":   price,
            "npv_meur":   kpis["npv_eur"] / 1e6,
            "lcoh":       kpis["lcoh_eur_per_kg"],
            "irr":        kpis["irr"],
            "rev_da":     kpis["da_export_revenue_yr_eur"] / 1e6,
            "rev_fcr":    kpis["fcr_revenue_yr_eur"]       / 1e6,
            "rev_h2":     kpis["h2_revenue_yr_eur"]        / 1e6,
            "ebitda":     (kpis["total_revenue_yr_eur"]
                           - kpis["total_opex_yr_eur"])     / 1e6,
        })

    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].bar(df["h2_price"], df["npv_meur"], width=0.4,
                color=["#B71C1C" if v < 0 else "#1B5E20" for v in df["npv_meur"]],
                alpha=0.85)
    axes[0].axhline(0, color="black", lw=0.8, alpha=0.4)
    for _, row in df.iterrows():
        irr = f"{row['irr']*100:.1f}%" if row["irr"] is not None else "N/A"
        axes[0].text(row["h2_price"], row["npv_meur"] + 0.5,
                     irr, ha="center", fontsize=8)
    axes[0].set_xlabel("H₂ price [EUR/kg]")
    axes[0].set_ylabel("NPV [MEUR]")
    axes[0].set_title("NPV sensitivity to H₂ price\n(labels = IRR)")
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].plot(df["h2_price"], df["lcoh"], "o-",
                 color="#0D47A1", lw=2, ms=8)
    for _, row in df.iterrows():
        axes[1].axvline(row["h2_price"], color="gray", ls=":", lw=0.5, alpha=0.5)
    axes[1].set_xlabel("H₂ price [EUR/kg]")
    axes[1].set_ylabel("LCOH [EUR/kg H₂]")
    axes[1].set_title("LCOH\n(constant — dispatch unchanged)")
    axes[1].grid(alpha=0.3)

    bottom = np.zeros(len(df))
    for col, label, color in [
        ("rev_da",  "DA export", "#1565C0"),
        ("rev_fcr", "FCR",       "#2E7D32"),
        ("rev_h2",  "H₂ sales",  "#F57F17"),
    ]:
        axes[2].bar(df["h2_price"], df[col].values, bottom=bottom,
                    width=0.4, label=label, color=color, alpha=0.85)
        bottom += df[col].values
    axes[2].plot(df["h2_price"], df["ebitda"], "ko--", lw=1.5,
                 ms=5, label="EBITDA (net)")
    axes[2].set_xlabel("H₂ price [EUR/kg]")
    axes[2].set_ylabel("MEUR/yr")
    axes[2].set_title("Revenue mix vs H₂ price")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"H₂ price sensitivity  —  electrolyzer {P_el_kw/1000:.0f} MW",
        fontsize=12, fontweight="medium",
    )
    plt.tight_layout()
    _save(fig, save_dir, "09_h2_sensitivity.png", show)
    return df