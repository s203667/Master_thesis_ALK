"""
sizing.py
=========
Electrolyzer capacity sizing study for the wind–hydrogen system.

Approach: parametric ratio sweep
---------------------------------
Run the existing rolling-horizon DA+RT optimisation for each candidate
electrolyzer capacity (expressed as a ratio r = P_el / P_wind_net), then
collect LCOH, NPV, IRR, EBITDA, FLH, and curtailment across the sweep.

Primary ranking metric: NPV
  — the only metric that correctly weighs both H2 revenue and grid export
    revenue against the full investment cost. LCOH alone would always push
    toward a smaller electrolyzer because it ignores grid revenue. EBITDA
    alone would push toward a larger one because it ignores CAPEX.

Optional: H2 price sensitivity sweep
  — re-runs the ratio sweep at multiple H2 prices and overlays the NPV
    curves, so you can see how the optimal ratio shifts with price.

Usage (from main.py)
--------------------
    from sizing import SizingStudy, SizingConfig

    sizing_cfg = SizingConfig(
        ratios           = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00],
        wind_capacity_kw = FARM_CAPACITY_KW * LOSS_FACTOR,
        econ_config_base = base_econ_cfg,
    )
    study     = SizingStudy(sizing_cfg)
    sizing_df = study.run(run_one_size)
    study.print_summary(sizing_df)
    study.plot(sizing_df)

    # Optional H2 price sensitivity:
    study.run_h2_sensitivity(
        run_one_size_fn  = run_one_size,
        h2_prices        = [3.0, 4.5, 6.0, 8.0],
        save_path        = "sizing_sensitivity.png",
    )
"""

from __future__ import annotations

import copy
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
from Economics import EconomicsAnalysis


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SizingConfig:
    """
    Parameters that control the sizing sweep.

    ratios : list of floats
        r = P_electrolyzer / P_wind_net to evaluate.
        Typical literature range is 0.05–1.0.
        Ratios > 1 are only meaningful if grid purchases are allowed.

    wind_capacity_kw : float
        Net installed wind capacity = FARM_CAPACITY_KW * LOSS_FACTOR.
        Do NOT use wind_df['wind_kW'].max() — that returns peak instantaneous
        output and is insensitive to loss factors.

    econ_config_base : EconomicsConfig
        Fully configured base config.  electrolyzer_capacity_kw is overridden
        each iteration; all other fields stay fixed.

    capex_scale_fn : callable (P_el_kw → EUR/kW), optional
        Economy-of-scale CAPEX curve.  If None, constant CAPEX is used.
        Example: lambda p: 1_000.0 * (p / 100_000.0) ** (-0.15)
    """
    ratios:            List[float]
    wind_capacity_kw:  float
    econ_config_base:  object
    capex_scale_fn:    Optional[Callable[[float], float]] = None


# ──────────────────────────────────────────────────────────────────────────────
# Study class
# ──────────────────────────────────────────────────────────────────────────────

class SizingStudy:

    def __init__(self, cfg: SizingConfig):
        self.cfg = cfg

    # ──────────────────────────────────────────────────────────────────────────
    # Core sweep
    # ──────────────────────────────────────────────────────────────────────────

    def run(
        self,
        run_one_size_fn: Callable[[float], List[Dict]],
        h2_price_override: Optional[float] = None,
        silent: bool = False,
    ) -> pd.DataFrame:
        """
        Execute the ratio sweep.

        Parameters
        ----------
        run_one_size_fn : callable
            Signature: run_one_size_fn(P_el_kw) -> List[Dict]
            Returns the per-day results_all list for that electrolyzer size.

        h2_price_override : float or None
            If set, replaces price_H2 in every day's result dict before
            computing economics.  Used by run_h2_sensitivity().

        silent : bool
            Suppress per-ratio progress output (used in sensitivity runs).

        Returns
        -------
        pd.DataFrame — one row per ratio.
        """
        from Economics import EconomicsAnalysis

        rows = []
        for r in self.cfg.ratios:
            P_el_kw = r * self.cfg.wind_capacity_kw

            if not silent:
                print(f"\n{'='*60}")
                print(f"  Sizing sweep: r = {r:.2f}  →  P_el = {P_el_kw/1000:.1f} MW")
                print(f"{'='*60}")

            results_all = run_one_size_fn(P_el_kw)

            # Optionally override H2 price in every day dict
            if h2_price_override is not None:
                results_all = _override_h2_price(results_all, h2_price_override)

            econ_cfg = self._make_econ_config(P_el_kw)
            econ     = EconomicsAnalysis(econ_cfg)
            kpis     = econ.compute(results_all)

            row = {
                "ratio":   r,
                "P_el_kw": P_el_kw,
                "P_el_mw": P_el_kw / 1000.0,
                **{k: kpis[k] for k in [
                    "lcoh_eur_per_kg",
                    "npv_eur",
                    "irr",
                    "payback_yr",
                    "total_revenue_yr_eur",
                    "total_opex_yr_eur",
                    "da_export_revenue_yr_eur",
                    "fcr_revenue_yr_eur",
                    "h2_revenue_yr_eur",
                    "imbalance_revenue_yr_eur",
                    "grid_cost_yr_eur",
                    "total_capex_eur",
                    "flh_electrolyzer_hr",
                    "curtailment_pct",
                    "total_h2_kg_yr",
                ]},
                "ebitda_yr_eur": kpis["total_revenue_yr_eur"] - kpis["total_opex_yr_eur"],
            }
            rows.append(row)
            if not silent:
                self._print_row(row)

        return pd.DataFrame(rows)

    # ──────────────────────────────────────────────────────────────────────────
    # H2 price sensitivity
    # ──────────────────────────────────────────────────────────────────────────

    def run_h2_sensitivity(
        self,
        run_one_size_fn: Callable[[float], List[Dict]],
        h2_prices: List[float],
        save_path: Optional[str] = None,
    ) -> Dict[float, pd.DataFrame]:
        """
        Re-run the ratio sweep at each H2 price and overlay NPV curves.

        The operational model (run_one_size_fn) is called once per ratio —
        the H2 price sensitivity is applied POST-HOC in economics only.
        This is an approximation: the operational dispatch would change
        slightly at different H2 prices, but the effect on sizing is small
        compared to the revenue effect.

        Parameters
        ----------
        h2_prices : list of floats  [EUR/kg]
        save_path : str or None

        Returns
        -------
        dict mapping h2_price → DataFrame (same schema as run())
        """
        print("\n" + "="*60)
        print("  H2 PRICE SENSITIVITY SWEEP")
        print("="*60)

        # Run the operational model once per ratio (price-agnostic dispatch)
        print("\nRunning operational model (dispatch is H2-price agnostic)...")
        base_results: Dict[float, List[Dict]] = {}
        for r in self.cfg.ratios:
            P_el_kw = r * self.cfg.wind_capacity_kw
            print(f"  r = {r:.2f}  ({P_el_kw/1000:.1f} MW)...", end=" ", flush=True)
            base_results[r] = run_one_size_fn(P_el_kw)
            print("done")

        # Sweep H2 prices using cached operational results
        sensitivity: Dict[float, pd.DataFrame] = {}
        for price in h2_prices:
            print(f"\n  H2 price = {price:.1f} EUR/kg")
            rows = []
            for r in self.cfg.ratios:
                P_el_kw     = r * self.cfg.wind_capacity_kw
                results_day = _override_h2_price(base_results[r], price)
                econ_cfg    = self._make_econ_config(P_el_kw)
                kpis        = EconomicsAnalysis(econ_cfg).compute(results_day)
                rows.append({
                    "ratio":             r,
                    "P_el_mw":           P_el_kw / 1000.0,
                    "npv_eur":           kpis["npv_eur"],
                    "lcoh_eur_per_kg":   kpis["lcoh_eur_per_kg"],
                    "ebitda_yr_eur":     kpis["total_revenue_yr_eur"] - kpis["total_opex_yr_eur"],
                    "flh_electrolyzer_hr": kpis["flh_electrolyzer_hr"],
                })
            sensitivity[price] = pd.DataFrame(rows)

        self._plot_sensitivity(sensitivity, save_path)
        return sensitivity

    # ──────────────────────────────────────────────────────────────────────────
    # Config helper
    # ──────────────────────────────────────────────────────────────────────────

    def _make_econ_config(self, P_el_kw: float):
        cfg = copy.deepcopy(self.cfg.econ_config_base)
        cfg.electrolyzer_capacity_kw = P_el_kw
        if self.cfg.capex_scale_fn is not None:
            cfg.capex_electrolyzer_eur_kw = self.cfg.capex_scale_fn(P_el_kw)
        return cfg

    # ──────────────────────────────────────────────────────────────────────────
    # Print helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _print_row(self, row: Dict) -> None:
        irr_str = f"{row['irr']*100:.1f}%" if row["irr"] is not None else "N/A"
        pb_str  = f"{row['payback_yr']:.1f}" if np.isfinite(row["payback_yr"]) else "∞"
        print(f"  LCOH        = {row['lcoh_eur_per_kg']:.2f} EUR/kg")
        print(f"  NPV         = {row['npv_eur']/1e6:.2f} MEUR   IRR = {irr_str}   Payback = {pb_str} yr")
        print(f"  EBITDA/yr   = {row['ebitda_yr_eur']/1e6:.2f} MEUR")
        print(f"  FLH         = {row['flh_electrolyzer_hr']:.0f} h/yr   Curtailment = {row['curtailment_pct']:.1f}%")
        print(f"  Revenue mix : DA {row['da_export_revenue_yr_eur']/1e6:.2f}  "
              f"FCR {row['fcr_revenue_yr_eur']/1e6:.2f}  "
              f"H2 {row['h2_revenue_yr_eur']/1e6:.2f}  "
              f"Imb {row['imbalance_revenue_yr_eur']/1e6:.2f}  "
              f"GridCost -{row['grid_cost_yr_eur']/1e6:.2f}  [MEUR/yr]")

    def print_summary(self, df: pd.DataFrame) -> None:
        """Ranked summary table with optima highlighted."""
        print("\n" + "=" * 80)
        print("  SIZING SWEEP SUMMARY  (primary objective: max NPV)")
        print("=" * 80)
        print(f"  {'r':>5}  {'P_el':>7}  {'LCOH':>8}  {'NPV':>9}  "
              f"{'IRR':>6}  {'Payback':>8}  {'FLH':>6}  {'Curt%':>6}  {'H2 t/yr':>8}")
        print(f"  {'':>5}  {'(MW)':>7}  {'€/kg':>8}  {'(MEUR)':>9}  "
              f"{'':>6}  {'(yr)':>8}  {'(h)':>6}  {'':>6}  {'':>8}")
        print("  " + "-"*76)

        # Sort by NPV descending for the display
        for _, row in df.sort_values("npv_eur", ascending=False).iterrows():
            irr_str = f"{row['irr']*100:.1f}%" if row["irr"] is not None else "  N/A"
            pb_str  = f"{row['payback_yr']:.1f}" if np.isfinite(row["payback_yr"]) else "    ∞"
            marker  = " ◄ max NPV" if row["ratio"] == df.loc[df["npv_eur"].idxmax(), "ratio"] else ""
            print(f"  {row['ratio']:>5.2f}  {row['P_el_mw']:>7.1f}  "
                  f"{row['lcoh_eur_per_kg']:>8.2f}  "
                  f"{row['npv_eur']/1e6:>9.1f}  "
                  f"{irr_str:>6}  "
                  f"{pb_str:>8}  "
                  f"{row['flh_electrolyzer_hr']:>6.0f}  "
                  f"{row['curtailment_pct']:>5.1f}%  "
                  f"{row['total_h2_kg_yr']/1000:>8.0f}"
                  f"{marker}")

        best_lcoh = df.loc[df["lcoh_eur_per_kg"].idxmin()]
        best_npv  = df.loc[df["npv_eur"].idxmax()]
        best_irr_idx = df["irr"].apply(lambda x: x if x is not None else -np.inf).idxmax()
        best_irr  = df.loc[best_irr_idx]

        print("\n  Optimal points:")
        print(f"    Max NPV   : r = {best_npv['ratio']:.2f}  ({best_npv['P_el_mw']:.1f} MW)"
              f"  →  NPV = {best_npv['npv_eur']/1e6:.1f} MEUR")
        print(f"    Min LCOH  : r = {best_lcoh['ratio']:.2f}  ({best_lcoh['P_el_mw']:.1f} MW)"
              f"  →  LCOH = {best_lcoh['lcoh_eur_per_kg']:.2f} EUR/kg")
        if best_irr["irr"] is not None:
            print(f"    Max IRR   : r = {best_irr['ratio']:.2f}  ({best_irr['P_el_mw']:.1f} MW)"
                  f"  →  IRR = {best_irr['irr']*100:.1f}%")
        if best_lcoh["ratio"] != best_npv["ratio"]:
            print(f"\n  Note: LCOH-optimal and NPV-optimal sizes differ — this is expected")
            print(f"  when grid export revenue competes with H2 revenue at high wind hours.")
        print("=" * 80)

    # ──────────────────────────────────────────────────────────────────────────
    # Plots
    # ──────────────────────────────────────────────────────────────────────────

    def plot(self, df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """
        Six-panel sizing trade-off plot:
          (1) NPV vs ratio             ← primary objective
          (2) LCOH vs ratio
          (3) EBITDA vs ratio
          (4) FLH + curtailment vs ratio
          (5) Revenue mix stacked bar
          (6) Annual H2 production
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            print("matplotlib not available — skipping plot.")
            return

        fig = plt.figure(figsize=(15, 10))
        gs  = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.38)

        r      = df["ratio"].values
        mw     = df["P_el_mw"].values
        xlabel = "r = P_el / P_wind"

        def add_mw_axis(ax):
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ticks = r[::max(1, len(r)//4)]
            ax2.set_xticks(ticks)
            ax2.set_xticklabels([f"{mw[np.where(r==t)[0][0]]:.0f} MW" for t in ticks],
                                 fontsize=8)
            return ax2

        def mark_optimum(ax, xs, ys, color):
            idx = np.argmax(ys) if ys.mean() > 0 else np.argmin(ys)
            ax.axvline(xs[idx], color=color, ls="--", lw=1, alpha=0.5)
            ax.scatter([xs[idx]], [ys[idx]], s=90, color=color, zorder=5,
                       edgecolors="white", linewidths=1.0)

        # ── 1. NPV (primary) ─────────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(r, df["npv_eur"]/1e6, "o-", color="#1B5E20", lw=2.5)
        ax1.axhline(0, color="black", lw=0.8, alpha=0.35)
        mark_optimum(ax1, r, df["npv_eur"].values, "#1B5E20")
        ax1.set_xlabel(xlabel, fontsize=9)
        ax1.set_ylabel("NPV  [MEUR]")
        ax1.set_title("Net present value  ★ primary", fontweight="bold")
        ax1.grid(alpha=0.3)
        add_mw_axis(ax1)

        # ── 2. LCOH ──────────────────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(r, df["lcoh_eur_per_kg"], "o-", color="#0D47A1", lw=2)
        mark_optimum(ax2, r, -df["lcoh_eur_per_kg"].values, "#0D47A1")
        ax2.set_xlabel(xlabel, fontsize=9)
        ax2.set_ylabel("LCOH  [EUR/kg H₂]")
        ax2.set_title("Levelised cost of hydrogen")
        ax2.grid(alpha=0.3)
        add_mw_axis(ax2)

        # ── 3. EBITDA ─────────────────────────────────────────────────────────
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(r, df["ebitda_yr_eur"]/1e6, "o-", color="#4A148C", lw=2)
        mark_optimum(ax3, r, df["ebitda_yr_eur"].values, "#4A148C")
        ax3.axhline(0, color="black", lw=0.8, alpha=0.35)
        ax3.set_xlabel(xlabel, fontsize=9)
        ax3.set_ylabel("EBITDA  [MEUR/yr]")
        ax3.set_title("Annual operating profit (pre-CAPEX)")
        ax3.grid(alpha=0.3)
        add_mw_axis(ax3)

        # ── 4. FLH + curtailment ─────────────────────────────────────────────
        ax4 = fig.add_subplot(gs[1, 0])
        c_flh, c_curt = "#6A1B9A", "#BF360C"
        ln1, = ax4.plot(r, df["flh_electrolyzer_hr"], "s-", color=c_flh, lw=2,
                        label="FLH [h/yr]")
        ax4.set_ylabel("Electrolyzer FLH  [h/yr]", color=c_flh)
        ax4.tick_params(axis="y", labelcolor=c_flh)
        ax4r = ax4.twinx()
        ln2, = ax4r.plot(r, df["curtailment_pct"], "^--", color=c_curt, lw=2,
                          label="Curtailment [%]")
        ax4r.set_ylabel("Wind curtailment  [%]", color=c_curt)
        ax4r.tick_params(axis="y", labelcolor=c_curt)
        ax4.set_xlabel(xlabel, fontsize=9)
        ax4.set_title("FLH and wind curtailment")
        ax4.grid(alpha=0.3)
        ax4.legend(handles=[ln1, ln2], fontsize=8, loc="center right")

        # ── 5. Revenue mix ────────────────────────────────────────────────────
        ax5 = fig.add_subplot(gs[1, 1])
        bar_w = min(0.04, (r[-1] - r[0]) / (len(r) * 1.5))
        rev_labels = ["DA export", "FCR", "H2 sales", "Imbalance"]
        rev_cols   = ["#1565C0", "#2E7D32", "#F57F17", "#6A1B9A"]
        rev_keys   = ["da_export_revenue_yr_eur", "fcr_revenue_yr_eur",
                      "h2_revenue_yr_eur", "imbalance_revenue_yr_eur"]
        bottom = np.zeros(len(r))
        for label, col, key in zip(rev_labels, rev_cols, rev_keys):
            vals = df[key].values / 1e6
            ax5.bar(r, vals, bottom=bottom, width=bar_w, label=label,
                    color=col, alpha=0.85)
            bottom += vals
        # Grid cost as negative bar
        ax5.bar(r, -df["grid_cost_yr_eur"].values / 1e6, width=bar_w,
                label="Grid cost", color="#B71C1C", alpha=0.7)
        ax5.axhline(0, color="black", lw=0.8, alpha=0.4)
        ax5.set_xlabel(xlabel, fontsize=9)
        ax5.set_ylabel("Revenue / cost  [MEUR/yr]")
        ax5.set_title("Revenue mix")
        ax5.legend(fontsize=7, loc="upper left", ncol=2)
        ax5.grid(alpha=0.3, axis="y")

        # ── 6. H2 production ──────────────────────────────────────────────────
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.bar(r, df["total_h2_kg_yr"]/1000, width=bar_w*1.1,
                color="#01579B", alpha=0.75, edgecolor="#01579B", lw=0.5)
        ax6.set_xlabel(xlabel, fontsize=9)
        ax6.set_ylabel("H₂ production  [t/yr]")
        ax6.set_title("Annual hydrogen production")
        ax6.grid(alpha=0.3, axis="y")

        # ── Literature optimum band ───────────────────────────────────────────
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
            ax.axvspan(0.05, 0.20, alpha=0.05, color="gray")

        # small annotation on panel 1
        ax1.text(0.125, ax1.get_ylim()[0] + 0.02*(ax1.get_ylim()[1]-ax1.get_ylim()[0]),
                 "lit. range\n(0.05–0.20)", ha="center", va="bottom",
                 fontsize=7, color="gray", alpha=0.8)

        fig.suptitle(
            f"Electrolyzer sizing study  —  wind farm {self.cfg.wind_capacity_kw/1000:.0f} MW net",
            fontsize=13, fontweight="medium",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Sizing plot saved to {save_path}")
        else:
            plt.show()

    def _plot_sensitivity(
        self,
        sensitivity: Dict[float, pd.DataFrame],
        save_path: Optional[str] = None,
    ) -> None:
        """Overlay NPV curves for each H2 price on a single axes."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
        except ImportError:
            print("matplotlib not available — skipping sensitivity plot.")
            return

        from Economics import EconomicsAnalysis  # noqa — imported here for lazy load

        prices = sorted(sensitivity.keys())
        cmap   = cm.get_cmap("plasma", len(prices))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for idx, price in enumerate(prices):
            df    = sensitivity[price]
            color = cmap(idx)
            label = f"{price:.1f} €/kg"

            # NPV
            axes[0].plot(df["ratio"], df["npv_eur"]/1e6, "o-",
                         color=color, lw=2, label=label)
            best = df.loc[df["npv_eur"].idxmax()]
            axes[0].scatter([best["ratio"]], [best["npv_eur"]/1e6],
                            s=80, color=color, zorder=5,
                            edgecolors="white", linewidths=1.0)

            # LCOH
            axes[1].plot(df["ratio"], df["lcoh_eur_per_kg"], "o-",
                         color=color, lw=2, label=label)

        axes[0].axhline(0, color="black", lw=0.8, alpha=0.35)
        axes[0].axvspan(0.05, 0.20, alpha=0.06, color="gray")
        axes[0].set_xlabel("r = P_el / P_wind")
        axes[0].set_ylabel("NPV  [MEUR]")
        axes[0].set_title("NPV vs ratio — H₂ price sensitivity\n(dots = optimum per price)")
        axes[0].legend(title="H₂ price", fontsize=9)
        axes[0].grid(alpha=0.3)

        axes[1].axvspan(0.05, 0.20, alpha=0.06, color="gray")
        axes[1].set_xlabel("r = P_el / P_wind")
        axes[1].set_ylabel("LCOH  [EUR/kg H₂]")
        axes[1].set_title("LCOH vs ratio — H₂ price sensitivity")
        axes[1].legend(title="H₂ price", fontsize=9)
        axes[1].grid(alpha=0.3)

        fig.suptitle(
            f"H₂ price sensitivity  —  wind farm {self.cfg.wind_capacity_kw/1000:.0f} MW net",
            fontsize=12, fontweight="medium",
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Sensitivity plot saved to {save_path}")
        else:
            plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _override_h2_price(results_all: List[Dict], price: float) -> List[Dict]:
    """
    Return a shallow copy of results_all with price_H2 replaced.
    Does NOT re-run the operational model — the dispatch is fixed.
    """
    import numpy as np
    out = []
    for day in results_all:
        d = dict(day)
        d["price_H2"] = np.full(len(d["price_H2"]), price)
        out.append(d)
    return out