"""
analysis.py
===========
Targeted analysis module answering the two core research questions:

  RQ1: When is it profitable to include an electrolyzer?
  RQ2: When is it profitable to participate in FCR?

Each function takes results_all (list of per-day dicts) and parameters,
and produces a self-contained plot + printed numbers.

Usage from main.py:
    from analysis import (
        electrolyzer_profitability,
        imbalance_decomposition,
        size_sensitivity,
        fcr_profitability,
        fcr_price_sensitivity,
        operational_overview,
    )
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Dict, Optional
import copy
import os


# ══════════════════════════════════════════════════════════════════════════════
# RQ1 — ELECTROLYZER PROFITABILITY
# ══════════════════════════════════════════════════════════════════════════════

def electrolyzer_profitability(
    results_all: List[Dict],
    h2_prices: List[float],
    econ_cfg_base,
    P_el_kw: float,
    save_dir: str = "figures/analysis/",
    show: bool = True,
):
    """
    RQ1: Opportunity cost vs H2 revenue across H2 prices.

    Plots:
      - Break-even H2 price per season (bar)
      - Revenue components at each H2 price (stacked bar)
      - Margin = H2 revenue - opportunity cost (line)
      - Electrolyzer on/off time split per season
    """
    from Economics import EconomicsAnalysis
    os.makedirs(save_dir, exist_ok=True)
    dt = 0.25

    # ── Per-day opportunity cost and H2 revenue ───────────────────────────────
    rows = []
    for day_idx, d in enumerate(results_all):
        P_elec    = np.asarray(d["P_elec_RT"])
        h_sell    = np.asarray(d["h_sell_RT"])
        price_DA  = np.asarray(d["price_DA"])
        price_H2  = np.asarray(d["price_H2"])
        z_on      = np.asarray(d["z_on_RT"])

        opp_cost  = float(np.sum(P_elec * price_DA * dt / 1000))   # EUR
        h2_rev    = float(np.sum(h_sell * price_H2 * dt))           # EUR
        h2_kg     = float(np.sum(h_sell * dt))                      # kg
        breakeven = opp_cost / h2_kg if h2_kg > 0 else np.nan

        rows.append({
            "day":       day_idx,
            "date":      d.get("date", str(day_idx)),
            "opp_cost":  opp_cost,
            "h2_rev":    h2_rev,
            "h2_kg":     h2_kg,
            "breakeven": breakeven,
            "util_pct":  100 * z_on.mean(),
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month

    # ── H2 price sensitivity on economics ────────────────────────────────────
    sens_rows = []
    for price in h2_prices:
        repriced = _override_h2_price(results_all, price)
        cfg = copy.deepcopy(econ_cfg_base)
        cfg.electrolyzer_capacity_kw = P_el_kw
        kpis = EconomicsAnalysis(cfg).compute(repriced)

        total_h2_kg = sum(np.sum(np.asarray(d["h_sell_RT"]) * dt)
                          for d in results_all) * (365 / len(results_all))
        total_opp = sum(
            float(np.sum(np.asarray(d["P_elec_RT"])
                         * np.asarray(d["price_DA"]) * dt / 1000))
            for d in results_all
        ) * (365 / len(results_all))

        sens_rows.append({
            "h2_price":     price,
            "npv_meur":     kpis["npv_eur"] / 1e6,
            "h2_rev_yr":    kpis["h2_revenue_yr_eur"] / 1e6,
            "da_rev_yr":    kpis["da_export_revenue_yr_eur"] / 1e6,
            "imb_rev_yr":   kpis["imbalance_revenue_yr_eur"] / 1e6,
            "opp_cost_yr":  total_opp / 1e6,
            "margin_yr":    (price * total_h2_kg - total_opp) / 1e6,
            "breakeven":    total_opp / total_h2_kg if total_h2_kg > 0 else np.nan,
        })

    sens = pd.DataFrame(sens_rows)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35)

    # Panel 1: Break-even price per day
    ax1 = fig.add_subplot(gs[0, 0])
    valid = df[df["breakeven"].notna()]
    ax1.bar(range(len(valid)), valid["breakeven"].values,
            color=["#B71C1C" if b > max(h2_prices) else "#1B5E20"
                   for b in valid["breakeven"].values],
            alpha=0.8, width=0.8)
    for price in h2_prices:
        ax1.axhline(price, ls="--", lw=1, alpha=0.6,
                    label=f"H₂ = {price} EUR/kg")
    ax1.set_xlabel("Day")
    ax1.set_ylabel("Break-even H₂ price [EUR/kg]")
    ax1.set_title("Daily break-even H₂ price\n(red = loss-making at any tested price)")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3, axis="y")

    # Panel 2: NPV sensitivity to H2 price
    ax2 = fig.add_subplot(gs[0, 1])
    colors = ["#B71C1C" if v < 0 else "#1B5E20" for v in sens["npv_meur"]]
    bars = ax2.bar(sens["h2_price"], sens["npv_meur"], width=0.4,
                   color=colors, alpha=0.85)
    ax2.axhline(0, color="black", lw=0.8, alpha=0.4)
    ax2.set_xlabel("H₂ price [EUR/kg]")
    ax2.set_ylabel("NPV [MEUR]")
    ax2.set_title("NPV vs H₂ price")
    ax2.grid(alpha=0.3, axis="y")

    # Panel 3: Margin (H2 revenue - opportunity cost)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(sens["h2_price"], sens["margin_yr"], "o-",
             color="#0D47A1", lw=2, ms=8)
    ax3.axhline(0, color="black", lw=0.8, alpha=0.4)
    ax3.fill_between(sens["h2_price"],
                     sens["margin_yr"],
                     where=sens["margin_yr"] >= 0,
                     alpha=0.15, color="#1B5E20", label="Profitable")
    ax3.fill_between(sens["h2_price"],
                     sens["margin_yr"],
                     where=sens["margin_yr"] < 0,
                     alpha=0.15, color="#B71C1C", label="Loss-making")
    be = sens["breakeven"].iloc[0]   # same for all (cost-side only)
    ax3.axvline(be, color="orange", ls="--", lw=1.5,
                label=f"Break-even: {be:.2f} EUR/kg")
    ax3.set_xlabel("H₂ price [EUR/kg]")
    ax3.set_ylabel("H₂ margin [MEUR/yr]  (revenue − opp. cost)")
    ax3.set_title("H₂ margin vs price")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    # Panel 4: Revenue mix at each H2 price
    ax4 = fig.add_subplot(gs[1, 1])
    bottom = np.zeros(len(sens))
    for col, label, color in [
        ("da_rev_yr",  "DA export",   "#1565C0"),
        ("h2_rev_yr",  "H₂ sales",    "#F57F17"),
        ("imb_rev_yr", "Imbalance",   "#6A1B9A"),
    ]:
        ax4.bar(sens["h2_price"], sens[col].values, bottom=bottom,
                width=0.4, label=label, color=color, alpha=0.85)
        bottom += sens[col].values
    ax4.plot(sens["h2_price"], sens["opp_cost_yr"], "r--o",
             lw=1.5, ms=6, label="Opp. cost")
    ax4.set_xlabel("H₂ price [EUR/kg]")
    ax4.set_ylabel("MEUR/yr")
    ax4.set_title("Revenue mix vs H₂ price")
    ax4.legend(fontsize=7)
    ax4.grid(alpha=0.3, axis="y")

    fig.suptitle("RQ1: Electrolyzer profitability — H₂ price sensitivity",
                 fontsize=12, fontweight="medium")
    plt.tight_layout()
    _save(fig, save_dir, "rq1_electrolyzer_profitability.png", show)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n=== RQ1: ELECTROLYZER PROFITABILITY ===")
    print(f"  Mean daily break-even H₂ price : {df['breakeven'].mean():.2f} EUR/kg")
    print(f"  Days loss-making at 4.5 EUR/kg : "
          f"{(df['breakeven'] > 4.5).sum()} / {len(df)}")
    print(f"\n  H₂ price  |  NPV (MEUR)  |  Margin (MEUR/yr)  |  Break-even")
    print(f"  {'─'*56}")
    for _, row in sens.iterrows():
        print(f"  {row['h2_price']:>7.1f}   |  {row['npv_meur']:>10.1f}  |"
              f"  {row['margin_yr']:>16.1f}  |  {row['breakeven']:>6.2f} EUR/kg")

    return df, sens


# ══════════════════════════════════════════════════════════════════════════════
# RQ1 — IMBALANCE DECOMPOSITION
# ══════════════════════════════════════════════════════════════════════════════

def imbalance_decomposition(
    results_all: List[Dict],
    P_max: float,
    save_dir: str = "figures/analysis/",
    show: bool = True,
):
    """
    RQ1: Where does system profit come from?
    Decomposes revenue into: H2 production, DA export, imbalance,
    grid purchases, and isolates the electrolyzer's contribution.

    Key question: Is profit driven by the electrolyzer or by wind export?
    """
    os.makedirs(save_dir, exist_ok=True)
    dt = 0.25

    rows = []
    for d in results_all:
        P_DA     = np.asarray(d["P_DA"])
        P_grid   = np.asarray(d["P_grid"])
        P_elec   = np.asarray(d["P_elec_RT"])
        P_curt   = np.asarray(d["P_curt_RT"])
        #P_buy    = np.asarray(d["P_buy_RT"])
        h_sell   = np.asarray(d["h_sell_RT"])
        wind     = np.asarray(d["wind_real"])
        price_DA = np.asarray(d["price_DA"])
        price_imb= np.asarray(d["price_imb"])
        price_H2 = np.asarray(d["price_H2"])

        imbalance = P_grid - P_DA

        # Revenue streams
        rev_da   = float(np.sum(P_DA  * price_DA  * dt / 1000))
        rev_h2   = float(np.sum(h_sell * price_H2 * dt))
        rev_imb  = float(np.sum(imbalance * price_imb * dt / 1000))
        #cost_buy = float(np.sum(P_buy * price_DA  * dt / 1000))
        net      = rev_da + rev_h2 + rev_imb #- cost_buy

        # Counterfactual: what if wind just exported everything (no electrolyzer)?
        # In this case P_DA = wind, P_grid = wind, no imbalance, no H2
        rev_wind_only = float(np.sum(wind * price_DA * dt / 1000))

        # Electrolyzer's marginal contribution
        elec_contribution = net - rev_wind_only

        # How much imbalance comes from electrolyzer rescheduling vs forecast error?
        # Forecast error imbalance: deviation even without electrolyzer rescheduling
        # (approximated as wind_actual - wind_forecast)
        wind_fc = np.asarray(d.get("wind_forecast", wind))
        forecast_error_imb = float(
            np.sum((wind - wind_fc) * price_imb * dt / 1000)
        )
        elec_imb = rev_imb - forecast_error_imb

        # P_buy contribution: does buying from grid help or hurt?
        #p_buy_slots = (P_buy > 10).sum()

        rows.append({
            "date":               d.get("date", ""),
            "rev_da":             rev_da,
            "rev_h2":             rev_h2,
            "rev_imb":            rev_imb,
            #"cost_buy":           cost_buy,
            "net":                net,
            "rev_wind_only":      rev_wind_only,
            "elec_contribution":  elec_contribution,
            "forecast_error_imb": forecast_error_imb,
            "elec_imb":           elec_imb,
            "curt_mwh":           float(np.sum(P_curt) * dt / 1000),
            #"buy_mwh":            float(np.sum(P_buy)  * dt / 1000),
            #"p_buy_slots":        p_buy_slots,
            "h2_kg":              float(np.sum(h_sell) * dt),
        })

    df = pd.DataFrame(rows)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    days = np.arange(len(df))

    # Panel 1: Revenue stack with wind-only counterfactual
    bottom = np.zeros(len(df))
    for col, label, color in [
        ("rev_da",  "DA export",  "#1565C0"),
        ("rev_h2",  "H₂ sales",   "#F57F17"),
        ("rev_imb", "Imbalance",  "#6A1B9A"),
    ]:
        vals = df[col].values
        axes[0].bar(days, vals, bottom=bottom, label=label,
                    color=color, alpha=0.85, width=0.8)
        bottom += vals
    #axes[0].bar(days, -df["cost_buy"].values, label="Grid cost",
               # color="#B71C1C", alpha=0.7, width=0.8)
    axes[0].plot(days, df["rev_wind_only"].values, "k--o",
                 ms=4, lw=1.5, label="Wind-only (no electrolyzer)")
    axes[0].axhline(0, color="black", lw=0.8, alpha=0.4)
    axes[0].set_title("Daily revenue vs wind-only counterfactual")
    axes[0].set_xlabel("Day")
    axes[0].set_ylabel("EUR")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(alpha=0.3, axis="y")

    # Panel 2: Electrolyzer marginal contribution
    colors = ["#1B5E20" if v >= 0 else "#B71C1C"
              for v in df["elec_contribution"].values]
    axes[1].bar(days, df["elec_contribution"].values, color=colors,
                alpha=0.8, width=0.8)
    axes[1].axhline(0, color="black", lw=0.8, alpha=0.4)
    axes[1].axhline(df["elec_contribution"].mean(), color="navy",
                    ls="--", lw=1.5,
                    label=f"Mean: {df['elec_contribution'].mean():,.0f} EUR/day")
    axes[1].set_title("Electrolyzer marginal contribution\n(vs wind-only)")
    axes[1].set_xlabel("Day")
    axes[1].set_ylabel("EUR/day")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")

    # Panel 3: Imbalance decomposition — forecast error vs electrolyzer
    axes[2].bar(days, df["forecast_error_imb"].values, label="Forecast error",
                color="#0D47A1", alpha=0.8, width=0.8)
    axes[2].bar(days, df["elec_imb"].values,
                bottom=df["forecast_error_imb"].values,
                label="Electrolyzer rescheduling", color="#F57F17", alpha=0.8, width=0.8)
    axes[2].axhline(0, color="black", lw=0.8, alpha=0.4)
    axes[2].set_title("Imbalance revenue decomposition")
    axes[2].set_xlabel("Day")
    axes[2].set_ylabel("EUR/day")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3, axis="y")

    # Panel 4: Grid purchase and curtailment volumes
    ax4b = axes[3].twinx()
    #axes[3].bar(days - 0.2, df["buy_mwh"].values, width=0.4,
     #           color="#E65100", alpha=0.8, label="Grid purchases [MWh]")
    ax4b.bar(days + 0.2, df["curt_mwh"].values, width=0.4,
             color="#37474F", alpha=0.6, label="Curtailment [MWh]")
    axes[3].set_ylabel("Grid purchases [MWh]", color="#E65100")
    ax4b.set_ylabel("Curtailment [MWh]", color="#37474F")
    axes[3].set_title("Grid purchases and curtailment")
    axes[3].set_xlabel("Day")
    lines1, labels1 = axes[3].get_legend_handles_labels()
    lines2, labels2 = ax4b.get_legend_handles_labels()
    axes[3].legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    axes[3].grid(alpha=0.3, axis="y")

    fig.suptitle("RQ1: Imbalance decomposition — where does profit come from?",
                 fontsize=12, fontweight="medium")
    plt.tight_layout()
    _save(fig, save_dir, "rq1_imbalance_decomposition.png", show)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n=== RQ1: IMBALANCE DECOMPOSITION ===")
    total = df["net"].sum()
    print(f"  Total net revenue        : {total:>12,.0f} EUR")
    print(f"  Wind-only counterfactual : {df['rev_wind_only'].sum():>12,.0f} EUR")
    print(f"  Electrolyzer adds        : {df['elec_contribution'].sum():>12,.0f} EUR  "
          f"({100*df['elec_contribution'].sum()/df['rev_wind_only'].sum():.1f}% uplift)")
    print(f"  Days electrolyzer adds value: "
          f"{(df['elec_contribution'] > 0).sum()} / {len(df)}")
    print(f"\n  Imbalance revenue total  : {df['rev_imb'].sum():>12,.0f} EUR")
    print(f"    of which forecast error: {df['forecast_error_imb'].sum():>12,.0f} EUR")
    print(f"    of which electrolyzer  : {df['elec_imb'].sum():>12,.0f} EUR")
    #print(f"\n  Total grid purchases     : {df['buy_mwh'].sum():>10.1f} MWh")
    print(f"  Total curtailment        : {df['curt_mwh'].sum():>10.1f} MWh")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# RQ1 — ELECTROLYZER SIZE SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════════

def size_sensitivity(
    run_one_size_fn,
    sizes_kw: List[float],
    econ_cfg_base,
    wind_capacity_kw: float,
    h2_price: float = 4.5,
    save_dir: str = "figures/analysis/",
    show: bool = True,
):
    """
    RQ1: How does profitability change with electrolyzer size?
    Runs the operational model at each size and compares KPIs.
    """
    from Economics import EconomicsAnalysis
    os.makedirs(save_dir, exist_ok=True)
    dt = 0.25

    rows = []
    for P_el_kw in sizes_kw:
        ratio = P_el_kw / wind_capacity_kw
        print(f"  Size: {P_el_kw/1000:.0f} MW  (r = {ratio:.2f})")

        results = run_one_size_fn(P_el_kw)

        # Override H2 price
        results = _override_h2_price(results, h2_price)

        cfg = copy.deepcopy(econ_cfg_base)
        cfg.electrolyzer_capacity_kw = P_el_kw
        kpis = EconomicsAnalysis(cfg).compute(results)

        # Break-even H2 price
        total_h2_kg  = sum(np.sum(np.asarray(d["h_sell_RT"]) * dt)
                           for d in results) * (365 / len(results))
        total_opp    = sum(
            float(np.sum(np.asarray(d["P_elec_RT"])
                         * np.asarray(d["price_DA"]) * dt / 1000))
            for d in results
        ) * (365 / len(results))
        breakeven = total_opp / total_h2_kg if total_h2_kg > 0 else np.nan

        rows.append({
            "P_el_mw":    P_el_kw / 1000,
            "ratio":      ratio,
            "npv_meur":   kpis["npv_eur"] / 1e6,
            "lcoh":       kpis["lcoh_eur_per_kg"],
            "flh":        kpis["flh_electrolyzer_hr"],
            "cf_pct":     kpis["flh_electrolyzer_hr"] / 8760 * 100,
            "curt_pct":   kpis["curtailment_pct"],
            "h2_kg_yr":   kpis["total_h2_kg_yr"] / 1000,   # tonnes
            "breakeven":  breakeven,
            "h2_rev_yr":  kpis["h2_revenue_yr_eur"] / 1e6,
            "da_rev_yr":  kpis["da_export_revenue_yr_eur"] / 1e6,
        })

    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    mw  = df["P_el_mw"].values

    def bar(ax, vals, ylabel, title, colors=None):
        c = colors or ["#1565C0"] * len(vals)
        ax.bar(mw, vals, width=mw[1]-mw[0] if len(mw) > 1 else 10,
               color=c, alpha=0.8)
        ax.set_xlabel("Electrolyzer size [MW]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
        ax.set_xticks(mw)
        ax.set_xticklabels([f"{m:.0f}" for m in mw])

    npv_colors = ["#1B5E20" if v >= 0 else "#B71C1C" for v in df["npv_meur"]]
    bar(axes[0], df["npv_meur"].values, "NPV [MEUR]", "NPV vs size", npv_colors)
    axes[0].axhline(0, color="black", lw=0.8, alpha=0.4)

    bar(axes[1], df["lcoh"].values, "LCOH [EUR/kg H₂]", "LCOH vs size")
    for ax in [axes[1]]:
        ax.axhline(h2_price, color="orange", ls="--", lw=1.5,
                   label=f"H₂ price = {h2_price}")
        ax.legend(fontsize=8)

    bar(axes[2], df["flh"].values, "FLH [h/yr]", "Electrolyzer FLH vs size")

    bar(axes[3], df["breakeven"].values, "EUR/kg",
        "Break-even H₂ price vs size")
    axes[3].axhline(h2_price, color="orange", ls="--", lw=1.5)

    bar(axes[4], df["curt_pct"].values, "Curtailment [%]",
        "Wind curtailment vs size")

    # Revenue mix
    bottom = np.zeros(len(df))
    for col, label, color in [
        ("da_rev_yr", "DA export", "#1565C0"),
        ("h2_rev_yr", "H₂ sales",  "#F57F17"),
    ]:
        w = mw[1] - mw[0] if len(mw) > 1 else 10
        axes[5].bar(mw, df[col].values, bottom=bottom,
                    width=w, label=label, color=color, alpha=0.85)
        bottom += df[col].values
    axes[5].set_xlabel("Electrolyzer size [MW]")
    axes[5].set_ylabel("MEUR/yr")
    axes[5].set_title("Revenue mix vs size")
    axes[5].legend(fontsize=8)
    axes[5].set_xticks(mw)
    axes[5].set_xticklabels([f"{m:.0f}" for m in mw])
    axes[5].grid(alpha=0.3, axis="y")

    fig.suptitle(f"RQ1: Electrolyzer size sensitivity  (H₂ = {h2_price} EUR/kg)",
                 fontsize=12, fontweight="medium")
    plt.tight_layout()
    _save(fig, save_dir, "rq1_size_sensitivity.png", show)

    print("\n=== RQ1: SIZE SENSITIVITY ===")
    print(f"  {'Size (MW)':>10}  {'Ratio':>6}  {'NPV (MEUR)':>11}  "
          f"{'LCOH':>7}  {'FLH':>6}  {'CF%':>5}  {'Breakeven':>10}")
    for _, r in df.iterrows():
        print(f"  {r['P_el_mw']:>10.0f}  {r['ratio']:>6.2f}  "
              f"{r['npv_meur']:>11.1f}  {r['lcoh']:>7.2f}  "
              f"{r['flh']:>6.0f}  {r['cf_pct']:>5.1f}%  "
              f"{r['breakeven']:>10.2f}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# RQ2 — FCR PROFITABILITY
# ══════════════════════════════════════════════════════════════════════════════

def fcr_profitability(
    results_all: List[Dict],
    P_max: float,
    P_min: float,
    save_dir: str = "figures/analysis/",
    show: bool = True,
):
    """
    RQ2: When is FCR participation worthwhile?

    Analyses per-block FCR decisions:
      - FCR revenue vs opportunity cost of headroom
      - Which blocks are accepted and at what prices
      - Relationship between FCR bid, electrolyzer load, and headroom
    """
    os.makedirs(save_dir, exist_ok=True)
    dt = 0.25
    dt_FCR = 4.0

    rows = []
    for d in results_all:
        DA_FCR   = np.asarray(d["DA_FCR"])          # (I,) per block
        price_FCR= np.asarray(d["price_FCR"])        # (I,)
        P_elec   = np.asarray(d["P_elec_RT"])        # (T,)
        price_DA = np.asarray(d["price_DA"])          # (T,)
        h_sell   = np.asarray(d["h_sell_RT"])         # (T,)
        price_H2 = np.asarray(d["price_H2"])          # (T,)
        A_s_mean = np.mean(d.get("A_s", [0.02]))      # kg/kWh (rough)

        I = len(DA_FCR)
        block_len = len(P_elec) // I   # timesteps per block (16 for 4h at 15min)

        for i in range(I):
            t_start = i * block_len
            t_end   = (i + 1) * block_len

            P_elec_block  = P_elec[t_start:t_end]
            price_DA_blk  = price_DA[t_start:t_end]
            price_H2_blk  = price_H2[t_start:t_end]

            r_FCR = float(DA_FCR[i])
            p_FCR = float(price_FCR[i])

            fcr_rev      = r_FCR * p_FCR * dt_FCR   # EUR

            # Headroom opportunity cost: if FCR=r, electrolyzer runs at
            # most P_max - r instead of P_max → lost hydrogen
            # Approximate: headroom × efficiency × H2 price × hours
            headroom_cost = (r_FCR * A_s_mean
                             * float(price_H2_blk.mean())
                             * dt_FCR)                # EUR

            # Headroom available (unused capacity above current load)
            mean_load    = float(P_elec_block.mean())
            headroom_avail = P_max - mean_load         # kW

            rows.append({
                "date":         d.get("date", ""),
                "block":        i,
                "r_FCR_kw":     r_FCR,
                "price_FCR":    p_FCR,
                "fcr_rev":      fcr_rev,
                "headroom_cost":headroom_cost,
                "net_fcr":      fcr_rev - headroom_cost,
                "mean_load_pct":100 * mean_load / P_max,
                "headroom_avail_kw": headroom_avail,
                "participated": r_FCR > 100,
            })

    df = pd.DataFrame(rows)

    # Break-even FCR price
    # At what FCR price does fcr_rev == headroom_cost?
    # r_FCR * price_FCR * 4 == r_FCR * A_s * H2_price * 4
    # price_FCR_breakeven = A_s * H2_price
    mean_A_s    = np.mean([d.get("A_s", [0.02])[0] for d in results_all])
    mean_H2_price = np.mean([np.mean(d["price_H2"]) for d in results_all])
    breakeven_FCR = mean_A_s * mean_H2_price   # EUR/kW

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    # Panel 1: FCR price distribution — participated vs not
    part   = df[df["participated"]]["price_FCR"]
    nopart = df[~df["participated"]]["price_FCR"]
    axes[0].hist(nopart, bins=30, alpha=0.6, color="#B71C1C",
                 label=f"Not participated (n={len(nopart)})")
    axes[0].hist(part, bins=30, alpha=0.7, color="#1B5E20",
                 label=f"Participated (n={len(part)})")
    axes[0].axvline(breakeven_FCR, color="orange", ls="--", lw=2,
                    label=f"Break-even: {breakeven_FCR:.4f} EUR/kW")
    axes[0].set_xlabel("FCR price [EUR/kW per 4h block]")
    axes[0].set_ylabel("Number of blocks")
    axes[0].set_title("FCR price distribution\n(participated vs not)")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    # Panel 2: Net FCR value per block
    axes[1].scatter(df["price_FCR"], df["net_fcr"],
                    c=df["participated"].map({True: "#1B5E20", False: "#B71C1C"}),
                    alpha=0.5, s=20)
    axes[1].axhline(0, color="black", lw=0.8, alpha=0.4)
    axes[1].axvline(breakeven_FCR, color="orange", ls="--", lw=1.5,
                    label=f"Break-even FCR price")
    axes[1].set_xlabel("FCR price [EUR/kW]")
    axes[1].set_ylabel("Net FCR value (rev − opp. cost) [EUR]")
    axes[1].set_title("Net FCR value per block\n(green = participated)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    # Panel 3: Headroom available vs FCR bid
    participated = df[df["participated"]]
    not_part     = df[~df["participated"]]
    axes[2].scatter(not_part["headroom_avail_kw"] / 1000,
                    not_part["price_FCR"],
                    alpha=0.3, s=15, color="#B71C1C", label="Not participated")
    if len(participated) > 0:
        axes[2].scatter(participated["headroom_avail_kw"] / 1000,
                        participated["price_FCR"],
                        alpha=0.7, s=30, color="#1B5E20", label="Participated",
                        marker="*")
    axes[2].axvline(0, color="black", lw=0.8, alpha=0.4)
    axes[2].set_xlabel("Available headroom [MW]")
    axes[2].set_ylabel("FCR price [EUR/kW]")
    axes[2].set_title("Headroom available vs FCR price\n(*= participated)")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    # Panel 4: FCR price multiplier sensitivity
    multipliers = [1, 2, 5, 10, 20, 50]
    fcr_revs = []
    for mult in multipliers:
        rev = sum(
            float(np.asarray(d["DA_FCR"])[i])
            * float(np.asarray(d["price_FCR"])[i]) * mult * dt_FCR
            for d in results_all
            for i in range(len(d["DA_FCR"]))
        ) * (365 / len(results_all)) / 1e6
        fcr_revs.append(rev)

    axes[3].plot(multipliers, fcr_revs, "o-", color="#2E7D32", lw=2, ms=8)
    axes[3].axhline(0, color="black", lw=0.8, alpha=0.4)
    axes[3].set_xlabel("FCR price multiplier (×current)")
    axes[3].set_ylabel("FCR revenue [MEUR/yr]")
    axes[3].set_title("FCR revenue vs price multiplier\n(at current participation levels)")
    axes[3].grid(alpha=0.3)
    axes[3].set_xscale("log")

    fig.suptitle("RQ2: FCR profitability analysis", fontsize=12, fontweight="medium")
    plt.tight_layout()
    _save(fig, save_dir, "rq2_fcr_profitability.png", show)

    print("\n=== RQ2: FCR PROFITABILITY ===")
    print(f"  FCR participation rate : {100*df['participated'].mean():.1f}% of blocks")
    print(f"  Mean FCR revenue/block : {df[df['participated']]['fcr_rev'].mean():.0f} EUR")
    print(f"  Break-even FCR price   : {breakeven_FCR:.4f} EUR/kW  "
          f"({breakeven_FCR*1000:.2f} EUR/MW)")
    print(f"  Current mean FCR price : "
          f"{df['price_FCR'].mean():.4f} EUR/kW")
    print(f"  Multiple needed for participation: "
          f"{breakeven_FCR / df['price_FCR'].mean():.1f}×")

    return df, breakeven_FCR


# ══════════════════════════════════════════════════════════════════════════════
# RQ2 — FCR PRICE SENSITIVITY (re-run with scaled prices)
# ══════════════════════════════════════════════════════════════════════════════

def fcr_price_sensitivity(
    run_one_day_fn,          # callable: (price_FCR_scaled) -> results_all
    price_FCR_base: np.ndarray,
    FCR_volume: np.ndarray,
    multipliers: List[float],
    econ_cfg_base,
    P_el_kw: float,
    save_dir: str = "figures/analysis/",
    show: bool = True,
):
    """
    RQ2: At what FCR price level does participation become worthwhile?
    Re-runs the operational model with scaled FCR prices.

    run_one_day_fn should accept a price_FCR array and return results_all.
    """
    from Economics import EconomicsAnalysis
    os.makedirs(save_dir, exist_ok=True)

    rows = []
    for mult in multipliers:
        print(f"  FCR price ×{mult:.1f}...")
        price_FCR_scaled = price_FCR_base * mult
        results = run_one_day_fn(price_FCR_scaled)

        cfg = copy.deepcopy(econ_cfg_base)
        cfg.electrolyzer_capacity_kw = P_el_kw
        kpis = EconomicsAnalysis(cfg).compute(results)

        fcr_participation = np.mean([
            (np.asarray(d["DA_FCR"]) > 100).mean()
            for d in results
        ])

        rows.append({
            "multiplier":      mult,
            "mean_price_FCR":  float(price_FCR_base.mean()) * mult,
            "fcr_rev_yr":      kpis["fcr_revenue_yr_eur"] / 1e6,
            "npv_meur":        kpis["npv_eur"] / 1e6,
            "fcr_part_pct":    100 * fcr_participation,
            "h2_rev_yr":       kpis["h2_revenue_yr_eur"] / 1e6,
        })

    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].plot(df["multiplier"], df["fcr_part_pct"], "o-",
                 color="#2E7D32", lw=2, ms=8)
    axes[0].set_xlabel("FCR price multiplier")
    axes[0].set_ylabel("FCR participation rate [%]")
    axes[0].set_title("FCR participation vs price")
    axes[0].grid(alpha=0.3)

    axes[1].plot(df["multiplier"], df["fcr_rev_yr"], "o-",
                 color="#1565C0", lw=2, ms=8, label="FCR revenue")
    axes[1].plot(df["multiplier"], df["h2_rev_yr"], "s--",
                 color="#F57F17", lw=1.5, ms=6, label="H₂ revenue")
    axes[1].set_xlabel("FCR price multiplier")
    axes[1].set_ylabel("Revenue [MEUR/yr]")
    axes[1].set_title("Revenue streams vs FCR price")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(df["multiplier"], df["npv_meur"], "o-",
                 color="#6A1B9A", lw=2, ms=8)
    axes[2].axhline(0, color="black", lw=0.8, alpha=0.4)
    axes[2].set_xlabel("FCR price multiplier")
    axes[2].set_ylabel("NPV [MEUR]")
    axes[2].set_title("NPV vs FCR price")
    axes[2].grid(alpha=0.3)

    fig.suptitle("RQ2: FCR price sensitivity", fontsize=12, fontweight="medium")
    plt.tight_layout()
    _save(fig, save_dir, "rq2_fcr_price_sensitivity.png", show)

    print("\n=== RQ2: FCR PRICE SENSITIVITY ===")
    print(f"  {'Mult':>5}  {'Price (EUR/kW)':>14}  "
          f"{'Part. rate':>11}  {'FCR rev (MEUR)':>14}  {'NPV (MEUR)':>11}")
    for _, r in df.iterrows():
        print(f"  {r['multiplier']:>5.1f}  {r['mean_price_FCR']:>14.4f}  "
              f"{r['fcr_part_pct']:>10.1f}%  {r['fcr_rev_yr']:>14.2f}  "
              f"{r['npv_meur']:>11.1f}")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _override_h2_price(results_all, price):
    out = []
    for d in results_all:
        day = dict(d)
        day["price_H2"] = np.full(len(d["price_H2"]), price)
        out.append(day)
    return out


def _save(fig, save_dir, filename, show):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)