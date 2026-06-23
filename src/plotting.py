fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.suptitle(f"H₂ price sensitivity: electrolyzer — {hi} vs {lo} EUR/kg\n"
             f"Days {START_DAY}–{START_DAY+N_DAYS-1}", fontsize=12)

# ── Panel 1: 4.5 EUR/kg load + difference from 3.0 EUR/kg ────────────────
ax = axes[0]

# Full load for 4.5 EUR/kg
ax.plot(t, P_elec[hi]/1000, color=colors[hi], lw=1.5,
        label=f"Electrolyzer load — {hi} EUR/kg")

# Shade where 3.0 runs less than 4.5 (most common case)
ax.fill_between(t, P_elec[lo]/1000, P_elec[hi]/1000,
                where=(P_elec[hi] > P_elec[lo]),
                color=colors[lo], alpha=0.4,
                label=f"Reduction at {lo} EUR/kg")

# Shade where 3.0 runs MORE than 4.5 (less common)
ax.fill_between(t, P_elec[hi]/1000, P_elec[lo]/1000,
                where=(P_elec[lo] > P_elec[hi]),
                color="green", alpha=0.4,
                label=f"Higher load at {lo} EUR/kg")

ax.axhline(139.406, color="black", lw=0.8, ls="--", label="P$_{max}$")
ax.axhline(20.911,  color="gray",  lw=0.8, ls=":",  label="P$_{min}$")
ax.set_ylabel("MW")
ax.set_ylim(bottom=0)
ax.set_title(f"Electrolyzer load at {hi} EUR/kg with reduction at {lo} EUR/kg shaded")
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)

# ── Panel 2: DA price vs break-even thresholds ───────────────────────────
ax = axes[1]
ax.plot(t, p_DA_r, color="black", lw=1.4, label="DA price")
for price in [hi, lo]:
    be = eff_mean * price * 1000
    ax.axhline(be, color=colors[price], lw=1.4, ls="--",
               label=f"Break-even {price} EUR/kg ({be:.0f} EUR/MWh)")
ax.fill_between(t, eff_mean*lo*1000, eff_mean*hi*1000,
                alpha=0.08, color="purple",
                label="Break-even range")
ax.set_ylabel("EUR/MWh")
ax.set_xlabel("Hour of window")
ax.set_title("DA price vs H₂ break-even thresholds")
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)

# ── Day separators ────────────────────────────────────────────────────────
for ax in axes:
    for d in range(1, N_DAYS):
        ax.axvline(d * 24, color="black", lw=0.8, ls=":")

plt.tight_layout()
plt.savefig("figures/h2_sens_electrolyzer.png", dpi=150, bbox_inches="tight")
plt.show()




######################################################

hi = 4.5
lo = 3.0

# ── Plot 1: Power flows ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.suptitle(f"H₂ price sensitivity: power flows — {hi} vs {lo} EUR/kg\n"
             f"Days {START_DAY}–{START_DAY+N_DAYS-1}", fontsize=12)

P_grid_hi = get_window(results[hi], "P_grid", START_DAY, N_DAYS)
P_grid_lo = get_window(results[lo], "P_grid", START_DAY, N_DAYS)

for ax, price, P_grid, P_da in [
    (axes[0], hi, P_grid_hi, P_DA[hi]),
    (axes[1], lo, P_grid_lo, P_DA[lo]),
]:
    ax.fill_between(t, 0, wind_r/1000,
                    alpha=0.15, color="steelblue", label="Wind realised")
    ax.plot(t, wind_f/1000, color="steelblue", lw=1.0, ls=":",
            label="Wind forecast")
    ax.plot(t, P_da/1000,   color=colors[price], lw=1.5, ls="--",
            label="P$_{DA}$ committed")
    ax.plot(t, P_grid/1000, color=colors[price], lw=1.2,
            label="P$_{grid}$ realised")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("MW")
    ax.set_title(f"H₂ = {price} EUR/kg")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

axes[1].set_xlabel("Hour of window")
for ax in axes:
    for d in range(1, N_DAYS):
        ax.axvline(d * 24, color="black", lw=0.8, ls=":")

plt.tight_layout()
plt.savefig("figures/h2_sens_powerflows.png", dpi=150, bbox_inches="tight")
plt.show()


# ── Plot 2: Electrolyzer load + prices ───────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.suptitle(f"H₂ price sensitivity: electrolyzer and prices — {hi} vs {lo} EUR/kg\n"
             f"Days {START_DAY}–{START_DAY+N_DAYS-1}", fontsize=12)

# Panel 1: electrolyzer load
ax = axes[0]
ax.plot(t, P_elec[hi]/1000, color=colors[hi], lw=1.5,
        label=f"H₂ = {hi} EUR/kg")
ax.plot(t, P_elec[lo]/1000, color=colors[lo], lw=1.5,
        label=f"H₂ = {lo} EUR/kg")
ax.fill_between(t, P_elec[lo]/1000, P_elec[hi]/1000,
                where=(P_elec[hi] > P_elec[lo]),
                alpha=0.15, color="purple",
                label=f"Higher load at {hi} EUR/kg")
ax.axhline(139.406, color="black", lw=0.8, ls="--", label="P$_{max}$")
ax.axhline(20.911,  color="gray",  lw=0.8, ls=":",  label="P$_{min}$")
ax.set_ylabel("MW")
ax.set_title("Electrolyzer load")
ax.set_ylim(bottom=0)
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)

# Panel 2: DA price vs break-evens
ax = axes[1]
ax.plot(t, p_DA_r, color="black", lw=1.4, label="DA price")
for price in [hi, lo]:
    be = eff_mean * price * 1000
    ax.axhline(be, color=colors[price], lw=1.4, ls="--",
               label=f"Break-even {price} EUR/kg ({be:.0f} EUR/MWh)")
ax.fill_between(t, eff_mean*lo*1000, eff_mean*hi*1000,
                alpha=0.08, color="purple",
                label="Break-even range")
ax.set_ylabel("EUR/MWh")
ax.set_xlabel("Hour of window")
ax.set_title("DA price vs H₂ break-even thresholds")
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)

for ax in axes:
    for d in range(1, N_DAYS):
        ax.axvline(d * 24, color="black", lw=0.8, ls=":")

plt.tight_layout()
#plt.savefig("figures/h2_sens_electrolyzer.png", dpi=150, bbox_inches="tight")
plt.show()