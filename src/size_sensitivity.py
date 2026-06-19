import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── define your runs here ─────────────────────────────────────────────────
# key = electrolyzer size in MW, value = pickle path
SIZE_RUNS = {
    0.39:  "results/results_h2_45_det_0.39cap.pkl",
    0.49:  "results/results_h2_45_det_0.49cap.pkl",
    0.61:  "results/results_h2_45_det_0.61cap.pkl",
    0.73:  "results/results_h2_45_det_year.pkl", #base
    0.81:  "results/results_h2_45_det_0.81cap.pkl",
    0.91:  "results/results_h2_45_det_0.91cap.pkl",
    1.1:  "results/results_h2_45_det_1.1cap.pkl",
    1.23: "results/results_h2_45_det_1.23cap.pkl",
    1.5: "results/results_h2_45_det_1.5cap.pkl",
    2.0: "results/results_h2_45_det_2.0cap.pkl",

}

BASE_P_MAX   = 190_000.0   # kW — your baseline electrolyzer
WIND_CAP_KW  = 190_000.0   # kW — wind farm (fixed across all runs)
PRICE_H2     = 4.5         # EUR/kg
EFF          = 0.020117326497756622       # kg/kWh
WACC         = 0.065
LIFETIME     = 20
CRF          = WACC * (1+WACC)**LIFETIME / ((1+WACC)**LIFETIME - 1)
CAPEX_ELEC   = 1_000.0     # EUR/kW
CAPEX_WIND   = 3_300.0     # EUR/kW
OPEX_ELEC    = 200.0       # EUR/kW/yr
OPEX_WIND    = 50          # EUR/kW/yr
dt           = 0.25
dt_FCR       = 4.0

def compute_metrics(results_all, P_max_kw):
    n_days = len(results_all)
    scale  = 365.0 / n_days

    P_elec = np.concatenate([d["P_elec_RT"]  for d in results_all])
    P_DA   = np.concatenate([d["P_DA"]        for d in results_all])
    P_grid = np.concatenate([d["P_grid"]      for d in results_all])
    P_buy  = np.concatenate([d["P_buy_RT"]    for d in results_all])
    P_curt = np.concatenate([d["P_curt_RT"]   for d in results_all])
    h_sell = np.concatenate([d["h_sell_RT"]   for d in results_all])
    wind   = np.concatenate([d["wind_real"]   for d in results_all])
    p_DA   = np.concatenate([d["price_DA"]    for d in results_all])
    p_imb  = np.concatenate([d["price_imb"]  for d in results_all])
    p_H2   = np.concatenate([d["price_H2"]   for d in results_all])
    DA_FCR = np.concatenate([d["DA_FCR"]      for d in results_all])
    p_FCR  = np.concatenate([d["price_FCR"]  for d in results_all])
    z_on   = np.concatenate([d["z_on_RT"]    for d in results_all])

    rev_DA   = float(np.sum(P_DA   * p_DA  * dt / 1000))
    rev_imb  = float(np.sum((P_grid - P_DA) * p_imb * dt / 1000))
    rev_H2   = float(np.sum(h_sell * p_H2  * dt))
    cost_buy = float(np.sum(P_buy  * p_DA  * dt / 1000))
    rev_FCR  = float(np.sum(DA_FCR * p_FCR * dt_FCR))
    net      = rev_DA + rev_imb + rev_H2 - cost_buy + rev_FCR

    h2_kg      = float(np.sum(h_sell) * dt)
    elec_gwh   = float(np.sum(P_elec) * dt / 1e6)
    wind_gwh   = float(np.sum(wind)   * dt / 1e6)
    curt_gwh   = float(np.sum(P_curt) * dt / 1e6)
    opp_cost   = float(np.sum(P_elec * p_DA * dt / 1000))
    breakeven  = opp_cost / max(h2_kg, 1)
    util       = float(np.mean(z_on) * 100)
    cap_factor = elec_gwh * 1e6 / P_max_kw / (n_days * 24) * 100
    flh        = elec_gwh * 1e6 / P_max_kw
    fcr_rate   = float(np.mean(DA_FCR > 0) * 100)

    # annualised economics
    capex_elec  = P_max_kw  * CAPEX_ELEC
    capex_wind  = WIND_CAP_KW * CAPEX_WIND
    capex_total = capex_elec + capex_wind
    opex_yr     = P_max_kw  * OPEX_ELEC + OPEX_WIND
    crf_cost_yr = capex_total * CRF
    total_cost  = crf_cost_yr + opex_yr

    net_yr      = net * scale
    h2_yr       = h2_kg * scale
    lcoh        = (total_cost + cost_buy * scale) / max(h2_yr, 1)
    lcoh_corrected = (
    (crf_cost_yr + opex_yr)          # total annualised cost
    - (rev_DA * scale + rev_imb * scale + rev_FCR * scale)   # subtract electricity revenues
    + cost_buy * scale                # add grid purchase cost
    ) / max(h2_yr, 1)

    print(f"  Standard LCOH: {lcoh:.2f}  Corrected LCOH: {lcoh_corrected:.2f} EUR/kg")
    lcoh = lcoh_corrected

    # NPV
    npv = sum(net_yr / (1+WACC)**y for y in range(1, LIFETIME+1)) - capex_total

    wind_to_h2  = elec_gwh / max(wind_gwh, 1e-9) * 100

    return {
        "P_max_MW":       P_max_kw / 1000,
        "ratio":          P_max_kw / WIND_CAP_KW,
        "net_eur_yr":     net_yr,
        "rev_H2_yr":      rev_H2  * scale,
        "rev_DA_yr":      rev_DA  * scale,
        "rev_imb_yr":     rev_imb * scale,
        "rev_FCR_yr":     rev_FCR * scale,
        "cost_buy_yr":    cost_buy * scale,
        "h2_kg_yr":       h2_yr,
        "elec_gwh_yr":    elec_gwh * scale,
        "wind_gwh_yr":    wind_gwh * scale,
        "curt_gwh_yr":    curt_gwh * scale,
        "curt_pct":       curt_gwh / max(wind_gwh, 1e-9) * 100,
        "util_pct":       util,
        "cap_factor_pct": cap_factor,
        "flh":            flh * scale,
        "fcr_rate_pct":   fcr_rate,
        "breakeven_h2":   breakeven,
        "lcoh":           lcoh,
        "npv":            npv,
        "wind_to_h2_pct": wind_to_h2,
    }

# ── load and compute ──────────────────────────────────────────────────────
rows = []
for ratio, path in SIZE_RUNS.items():
    with open(path, "rb") as f:
        results = pickle.load(f)
    P_max_kw = BASE_P_MAX * ratio
    m = compute_metrics(results, P_max_kw)
    rows.append(m)
    print(f"  ratio={ratio:.2f}  P_max={m['P_max_MW']:.0f} MW  "
          f"net={m['net_eur_yr']/1e6:.1f} MEUR/yr  "
          f"LCOH={m['lcoh']:.2f} EUR/kg  CF={m['cap_factor_pct']:.1f}%")

df = pd.DataFrame(rows).sort_values("ratio").reset_index(drop=True)
print("\n", df[["ratio","P_max_MW","net_eur_yr","lcoh","cap_factor_pct",
               "util_pct","curt_pct","breakeven_h2","npv"]].to_string(index=False))

# ── plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Electrolyzer sizing sensitivity — deterministic full-year", fontsize=13)

x     = df["ratio"].values
x_mw  = df["P_max_MW"].values
xl    = "Electrolyzer / wind capacity ratio"

# 1. Net revenue + LCOH
ax = axes[0, 0]
ax.plot(x, df["net_eur_yr"]/1e6, "o-", color="steelblue", lw=2, label="Net revenue")
ax.set_xlabel(xl); ax.set_ylabel("MEUR/yr"); ax.grid(alpha=0.3)
ax.set_title("Net annual revenue")
ax2 = ax.twinx()
ax2.plot(x, df["lcoh"], "s--", color="darkorange", lw=2, label="LCOH")
ax2.set_ylabel("LCOH (EUR/kg)", color="darkorange")
ax2.tick_params(axis="y", labelcolor="darkorange")
ax2.axhline(PRICE_H2, color="red", lw=1, ls=":", label=f"H₂ price ({PRICE_H2} EUR/kg)")
lines1, lab1 = ax.get_legend_handles_labels()
lines2, lab2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, lab1+lab2, fontsize=8)

# 2. Capacity factor + utilisation
ax = axes[0, 1]
ax.plot(x, df["cap_factor_pct"], "o-", color="steelblue", lw=2, label="Capacity factor")
ax.plot(x, df["util_pct"],       "s--", color="orangered",  lw=2, label="Utilisation")
ax.set_xlabel(xl); ax.set_ylabel("%"); ax.grid(alpha=0.3)
ax.set_title("Capacity factor and utilisation")
ax.legend(fontsize=8)

# 3. NPV
ax = axes[0, 2]
ax.bar(x, df["npv"]/1e6, width=0.08, color=["green" if v > 0 else "red"
       for v in df["npv"]], alpha=0.8)
ax.axhline(0, color="black", lw=1)
ax.set_xlabel(xl); ax.set_ylabel("MEUR"); ax.grid(axis="y", alpha=0.3)
ax.set_title("NPV")
for i, (xi, yi) in enumerate(zip(x, df["npv"]/1e6)):
    ax.text(xi, yi + (2 if yi > 0 else -5), f"{yi:.0f}", ha="center", fontsize=8)

# 4. Revenue breakdown stacked bar
ax = axes[1, 0]
w = 0.06
ax.bar(x, df["rev_H2_yr"]/1e6,   w, label="H₂",         color="steelblue")
ax.bar(x, df["rev_DA_yr"]/1e6,   w, bottom=df["rev_H2_yr"]/1e6,
       label="DA export", color="royalblue")
ax.bar(x, df["rev_imb_yr"]/1e6,  w,
       bottom=(df["rev_H2_yr"]+df["rev_DA_yr"])/1e6,
       label="Imbalance",  color="cornflowerblue")
ax.bar(x, df["rev_FCR_yr"]/1e6,  w,
       bottom=(df["rev_H2_yr"]+df["rev_DA_yr"]+df["rev_imb_yr"])/1e6,
       label="FCR",        color="lightsteelblue")
ax.bar(x, -df["cost_buy_yr"]/1e6, w, label="Grid cost", color="salmon", alpha=0.8)
ax.set_xlabel(xl); ax.set_ylabel("MEUR/yr")
ax.set_title("Revenue breakdown"); ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)

# 5. Curtailment + wind-to-H2
ax = axes[1, 1]
ax.plot(x, df["curt_pct"],      "o-", color="red",       lw=2, label="Curtailment %")
ax.plot(x, df["wind_to_h2_pct"],"s--", color="steelblue", lw=2, label="Wind→H₂ %")
ax.set_xlabel(xl); ax.set_ylabel("%"); ax.grid(alpha=0.3)
ax.set_title("Wind utilisation")
ax.legend(fontsize=8)

# 6. Break-even + FCR participation
ax = axes[1, 2]
ax.plot(x, df["breakeven_h2"], "o-", color="steelblue", lw=2, label="Operational break-even")
ax.axhline(PRICE_H2, color="red", lw=1, ls="--", label=f"Selling price ({PRICE_H2} EUR/kg)")
ax.set_xlabel(xl); ax.set_ylabel("EUR/kg"); ax.grid(alpha=0.3)
ax.set_title("H₂ break-even price")
ax2 = ax.twinx()
ax2.plot(x, df["fcr_rate_pct"], "s--", color="darkorange", lw=2, label="FCR rate (%)")
ax2.set_ylabel("FCR participation (%)", color="darkorange")
ax2.tick_params(axis="y", labelcolor="darkorange")
lines1, lab1 = ax.get_legend_handles_labels()
lines2, lab2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, lab1+lab2, fontsize=8)

plt.tight_layout()
plt.savefig("figures/sizing_sensitivity.png", dpi=150, bbox_inches="tight")
plt.show()