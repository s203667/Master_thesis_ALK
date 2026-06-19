import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

def plot_deterministic_analysis(results_all, P_max, P_min, start_date="2026-01-07"):
    """
    Four analysis figures for the deterministic full-year run:
    1. Monthly revenue breakdown
    2. Duration curves
    3. Price vs dispatch scatter
    4. Typical operational day comparison (high wind vs low wind)
    """
    dt = 0.25
    n_days = len(results_all)

    # ── rebuild arrays ────────────────────────────────────────────────────
    P_elec  = np.concatenate([d["P_elec_RT"]  for d in results_all])
    P_DA    = np.concatenate([d["P_DA"]        for d in results_all])
    P_grid  = np.concatenate([d["P_grid"]      for d in results_all])
    P_buy   = np.concatenate([d["P_buy_RT"]    for d in results_all])
    P_curt  = np.concatenate([d["P_curt_RT"]   for d in results_all])
    h_sell  = np.concatenate([d["h_sell_RT"]   for d in results_all])
    wind    = np.concatenate([d["wind_real"]    for d in results_all])
    p_DA    = np.concatenate([d["price_DA"]     for d in results_all])
    p_imb   = np.concatenate([d["price_imb"]   for d in results_all])
    p_H2    = np.concatenate([d["price_H2"]     for d in results_all])
    DA_FCR  = np.concatenate([d["DA_FCR"]       for d in results_all])
    p_FCR   = np.concatenate([d["price_FCR"]    for d in results_all])
    z_on    = np.concatenate([d["z_on_RT"]      for d in results_all])

    dates = pd.date_range(start=start_date, periods=n_days, freq="D")

    # ── daily aggregates ──────────────────────────────────────────────────
    def daily(arr, agg="sum"):
        vals = arr.reshape(n_days, 96)
        return vals.sum(axis=1) if agg == "sum" else vals.mean(axis=1)

    rev_DA_day   = daily(P_DA   * p_DA  * dt / 1000)
    rev_imb_day  = daily((P_grid - P_DA) * p_imb * dt / 1000)
    rev_H2_day   = daily(h_sell * p_H2  * dt)
    cost_buy_day = daily(P_buy  * p_DA  * dt / 1000)
    rev_FCR_day  = np.array([
        np.sum(d["DA_FCR"] * d["price_FCR"] * 4) for d in results_all
    ])
    net_day = rev_DA_day + rev_imb_day + rev_H2_day - cost_buy_day + rev_FCR_day

    # ── monthly aggregates ────────────────────────────────────────────────
    df_daily = pd.DataFrame({
        "date":     dates,
        "rev_DA":   rev_DA_day,
        "rev_imb":  rev_imb_day,
        "rev_H2":   rev_H2_day,
        "cost_buy": cost_buy_day,
        "rev_FCR":  rev_FCR_day,
        "net":      net_day,
        "wind_GWh": daily(wind) * dt / 1e6,
        "util":     daily(z_on, "mean") * 100,
        "elec_MW":  daily(P_elec, "mean") / 1000,
    })
    df_daily.set_index("date", inplace=True)
    df_monthly = df_daily.resample("M").sum()
    df_monthly_mean = df_daily.resample("M").mean()

    # ════════════════════════════════════════════════════════════════════
    # FIGURE 1: Monthly revenue breakdown
    # ════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle("Monthly revenue breakdown — deterministic model", fontsize=13)

    months = df_monthly.index.strftime("%b")
    x = np.arange(len(months))
    w = 0.6

    ax = axes[0]
    ax.bar(x, df_monthly["rev_H2"]/1e6,    w, label="H₂ revenue",         color="steelblue")
    ax.bar(x, df_monthly["rev_DA"]/1e6,    w, bottom=df_monthly["rev_H2"]/1e6,
           label="DA export",              color="royalblue")
    ax.bar(x, df_monthly["rev_imb"]/1e6,   w,
           bottom=(df_monthly["rev_H2"]+df_monthly["rev_DA"])/1e6,
           label="Imbalance",              color="cornflowerblue")
    ax.bar(x, df_monthly["rev_FCR"]/1e6,   w,
           bottom=(df_monthly["rev_H2"]+df_monthly["rev_DA"]+df_monthly["rev_imb"])/1e6,
           label="FCR",                   color="lightsteelblue")
    ax.bar(x, -df_monthly["cost_buy"]/1e6, w, label="Grid purchase (cost)",
           color="salmon", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(months)
    ax.set_ylabel("MEUR"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Monthly revenue components")

    ax = axes[1]
    ax.bar(x, df_monthly["net"]/1e6, w, color="steelblue", alpha=0.8)
    ax2 = ax.twinx()
    ax2.plot(x, df_monthly_mean["util"], color="darkorange", lw=2,
             marker="o", ms=4, label="Mean utilisation (%)")
    ax2.plot(x, df_monthly_mean["wind_GWh"]*1e3/P_max/24/30*100,
             color="green", lw=2, ls="--", marker="s", ms=4,
             label="Wind capacity factor (%)")
    ax.set_xticks(x); ax.set_xticklabels(months)
    ax.set_ylabel("Net revenue (MEUR)")
    ax2.set_ylabel("(%)", color="darkorange")
    ax2.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Monthly net revenue with electrolyzer utilisation and wind capacity factor")

    plt.tight_layout()
    plt.savefig("figures/det_monthly_revenue.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ════════════════════════════════════════════════════════════════════
    # FIGURE 2: Duration curves
    # ════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Duration curves — full year deterministic", fontsize=13)

    pct = np.linspace(0, 100, len(P_elec))

    ax = axes[0]
    ax.plot(pct, np.sort(P_elec/1000)[::-1], color="orangered", lw=1.5)
    ax.axhline(P_max/1000, color="red",  lw=1, ls="--", label=f"P_max ({P_max/1000:.0f} MW)")
    ax.axhline(P_min/1000, color="gray", lw=1, ls="--", label=f"P_min ({P_min/1000:.0f} MW)")
    ax.axhline(0,          color="black",lw=0.5)
    ax.set_xlabel("% of time exceeded"); ax.set_ylabel("MW")
    ax.set_title("Electrolyzer power")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(pct, np.sort(wind/1000)[::-1], color="steelblue", lw=1.5)
    ax.axhline(wind.max()/1000, color="red", lw=1, ls="--",
               label=f"Peak ({wind.max()/1000:.0f} MW)")
    ax.set_xlabel("% of time exceeded"); ax.set_ylabel("MW")
    ax.set_title("Wind power")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(pct, np.sort(p_DA)[::-1], color="gray", lw=1.5, label="DA price")
    ax.plot(pct, np.sort(p_imb)[::-1], color="crimson", lw=1.2,
            ls="--", label="Imbalance price")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("% of time exceeded"); ax.set_ylabel("EUR/MWh")
    ax.set_title("Electricity prices")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/det_duration_curves.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ════════════════════════════════════════════════════════════════════
    # FIGURE 3: Price vs dispatch scatter + marginal cost line
    # ════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Price-dispatch relationship — deterministic model", fontsize=13)

    # marginal cost of H2 from grid: price at which buying grid power for H2 breaks even
    # break-even: price_DA [EUR/MWh] × (1/eff [MWh/kg]) = price_H2 [EUR/kg]
    eff_mean     = 0.020          # kg/kWh = 20 kg/MWh
    marginal_cost = p_H2[0] / eff_mean  # EUR/MWh → at this price, H2 from grid = breakeven
    print(f"Grid purchase break-even DA price: {marginal_cost:.1f} EUR/MWh")

    ax = axes[0]
    sc = ax.scatter(p_DA, P_elec/1000, c=wind/1000, cmap="Blues",
                    alpha=0.15, s=4, vmin=0, vmax=P_max/1000)
    plt.colorbar(sc, ax=ax, label="Wind power (MW)")
    ax.axvline(marginal_cost, color="red", lw=1.5, ls="--",
               label=f"Grid purchase break-even ({marginal_cost:.0f} EUR/MWh)")
    ax.set_xlabel("DA electricity price (EUR/MWh)")
    ax.set_ylabel("Electrolyzer power (MW)")
    ax.set_title("Electrolyzer dispatch vs DA price\n(colour = wind availability)")
    ax.legend(fontsize=8); ax.grid(alpha=0.2)

    ax = axes[1]
    sc2 = ax.scatter(p_DA, P_buy/1000, c=P_elec/1000, cmap="Oranges",
                     alpha=0.3, s=6, vmin=0, vmax=P_max/1000)
    plt.colorbar(sc2, ax=ax, label="Electrolyzer power (MW)")
    ax.axvline(marginal_cost, color="red", lw=1.5, ls="--",
               label=f"Break-even ({marginal_cost:.0f} EUR/MWh)")
    ax.set_xlabel("DA electricity price (EUR/MWh)")
    ax.set_ylabel("Grid import (MW)")
    ax.set_title("Grid purchases vs DA price\n(colour = electrolyzer load)")
    ax.legend(fontsize=8); ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig("figures/det_price_dispatch.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ════════════════════════════════════════════════════════════════════
    # FIGURE 4: Typical operational day — high wind vs low wind
    # ════════════════════════════════════════════════════════════════════
    daily_wind = np.array([d["wind_real"].mean() for d in results_all])
    high_wind_day = int(np.argmax(daily_wind))
    low_wind_day  = int(np.where(daily_wind > 0)[0][np.argmin(daily_wind[daily_wind > 0])])

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex="col")
    fig.suptitle("Operational behaviour — high wind day vs low wind day", fontsize=13)
    t = np.arange(96) * dt

    for col, (day_idx, day_label) in enumerate(
            [(high_wind_day, f"High wind day (Day {high_wind_day+1})"),
             (low_wind_day,  f"Low wind day  (Day {low_wind_day+1})")]):
        d = results_all[day_idx]

        ax = axes[0, col]
        ax.fill_between(t, 0, d["wind_real"]/1000,
                        alpha=0.2, color="steelblue", label="Wind")
        ax.plot(t, d["P_elec_RT"]/1000,  color="orangered",  lw=1.5, label="Electrolyzer")
        ax.plot(t, d["P_DA"]/1000,       color="navy",       lw=1.2, ls="--", label="P_DA")
        ax.plot(t, d["P_grid"]/1000,     color="royalblue",  lw=1.2, label="P_grid")
        ax.plot(t, d["P_buy_RT"]/1000,   color="crimson",    lw=1.0, ls=":", label="P_buy")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_ylabel("MW"); ax.set_title(day_label)
        ax.legend(fontsize=7, ncol=3); ax.grid(alpha=0.3)

        ax = axes[1, col]
        ax.plot(t, d["price_DA"],  color="gray",    lw=1.5, label="DA price")
        ax.plot(t, d["price_imb"], color="crimson", lw=1.2, label="Imbalance price")
        ax.axhline(marginal_cost,  color="red",    lw=1.0, ls="--",
                   label=f"Break-even ({marginal_cost:.0f} EUR/MWh)")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel("EUR/MWh"); ax.set_xlabel("Hour of day")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/det_typical_days.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\nAll figures saved to figures/")


# call it
plot_deterministic_analysis(results_all, P_max=P_max, P_min=P_min,
                            start_date="2026-01-07")