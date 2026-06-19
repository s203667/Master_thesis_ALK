"""
economics.py
============
Economic analysis for the wind–hydrogen system.

Revenue attribution:
  - DA stage  : DA grid export revenue  +  FCR capacity revenue
  - RT stage  : Hydrogen sale revenue   +  Imbalance settlement revenue
                +  Grid purchase cost (negative revenue / cost)

Metrics:
  - LCOH  (Levelised Cost of Hydrogen)      [EUR / kg_H2]
  - LCOE  (Levelised Cost of Electricity)   [EUR / MWh]
  - NPV   (Net Present Value)               [EUR]
  - IRR   (Internal Rate of Return)         [fraction, e.g. 0.12 = 12%]

All system-size parameters (electrolyzer_capacity_kw, wind_capacity_kw,
storage_capacity_kg) are passed in from main.py — no default values are
hard-coded here so there is a single source of truth.

Usage (from main.py)
--------------------
    from economics import EconomicsAnalysis, EconomicsConfig

    cfg = EconomicsConfig(
        # ── system sizes: derive from data / model, never hard-code ──
        electrolyzer_capacity_kw    = P_max,
        wind_capacity_kw            = wind_df['wind_kW'].max(),   # peak observed [kW]
        storage_capacity_kg         = storage_max,

        # ── capital costs ─────────────────────────────────────────────
        capex_electrolyzer_eur_kw   = 1_000.0,
        capex_wind_eur_kw           = 2_000.0,
        capex_storage_eur_kg        = 500.0,

        # ── operating costs ───────────────────────────────────────────
        opex_electrolyzer_eur_kw_yr = 20.0,
        opex_wind_eur_kw_yr         = 50.0,

        # ── financial ─────────────────────────────────────────────────
        discount_rate               = 0.08,
        project_lifetime_yr         = 20,
        simulation_years            = num_days / 365.0,
        tax_rate                    = 0.22,
    )

    econ = EconomicsAnalysis(cfg)
    econ.compute(results_all)
    econ.print_summary()
    econ_results = econ.to_dict()
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Configuration dataclass  —  no defaults on system sizes, must come from main
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EconomicsConfig:
    """
    All techno-economic parameters.

    System sizes (electrolyzer_capacity_kw, wind_capacity_kw,
    storage_capacity_kg) have NO defaults — they must be supplied from
    main.py so there is exactly one place where these numbers live.

    CAPEX/OPEX/financial parameters do have sensible defaults that you
    can override as needed.
    """

    # ── System sizes (NO defaults — must be set in main.py) ──────────────────
    electrolyzer_capacity_kw:    float   # rated electrolyzer power      [kW]
    wind_capacity_kw:            float   # installed (net) wind capacity  [kW]
    storage_capacity_kg:         float   # H2 storage capacity           [kg]

    # ── CAPEX  [EUR/kW or EUR/kg] ─────────────────────────────────────────────
    capex_electrolyzer_eur_kw:   float = 0.0
    capex_wind_eur_kw:           float = 2_000.0
    capex_storage_eur_kg:        float = 0.0

    # ── OPEX   [EUR/(kW·yr)] ──────────────────────────────────────────────────
    opex_electrolyzer_eur_kw_yr: float = 150.0
    opex_wind_eur_kw_yr:         float = 0.0

    # ── Financial ─────────────────────────────────────────────────────────────
    discount_rate:               float = 0.08
    project_lifetime_yr:         int   = 20
    simulation_years:            float = 1.0   # set to num_days / 365 in main
    tax_rate:                    float = 0.22


# ──────────────────────────────────────────────────────────────────────────────
# Core analysis class
# ──────────────────────────────────────────────────────────────────────────────

class EconomicsAnalysis:
    """
    Compute economic KPIs from rolling-horizon optimisation results.

    Parameters
    ----------
    config : EconomicsConfig
        All techno-economic parameters.  System sizes must come from main.py.
    """

    def __init__(self, config: EconomicsConfig):
        self.cfg = config
        self._results: Dict = {}

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def compute(
        self,
        results_all: List[Dict],
        dt: float = 0.25,
    ) -> Dict:
        """
        Compute all economic KPIs.

        Parameters
        ----------
        results_all : list of dicts
            One entry per simulated day.  Required keys per dict:
              'P_DA'       – array [kW]      DA export schedule
              'P_grid'     – array [kW]      RT actual grid export
              'h_sell_RT'  – array [kg/slot] hydrogen sold each slot
              'DA_FCR'     – array [kW]      FCR capacity per 4-h block
              'imbalance'  – array [kW]      P_grid − P_DA
              'wind_real'  – array [kW]      actual wind power
              'P_curt_RT'  – array [kW]      curtailed wind power
              'P_elec_RT'  – array [kW]      electrolyzer consumption
              'P_buy_RT'   – array [kW]      grid electricity purchased
              'price_DA'   – array [EUR/MWh] realised DA price
              'price_imb'  – array [EUR/MWh] imbalance settlement price
              'price_H2'   – array [EUR/kg]  hydrogen price
              'price_FCR'  – array [EUR/kW]  FCR price per block

        dt : float
            Time-step length in hours (default 0.25 for 15-min slots).

        Returns
        -------
        dict  (also stored as self._results)
        """

        # ── 1. Accumulate over all simulated days ─────────────────────────────

        da_export_revenue_sim  = 0.0
        fcr_revenue_sim        = 0.0
        h2_revenue_sim         = 0.0
        imbalance_revenue_sim  = 0.0
        grid_cost_sim          = 0.0   # cost of buying electricity from grid

        total_h2_kg_sim        = 0.0
        total_wind_kwh_sim     = 0.0   # available wind energy (before curtailment)
        total_curt_kwh_sim     = 0.0   # curtailed wind energy
        total_elec_kwh_sim     = 0.0   # electrolyzer consumption

        dt_FCR = 4.0   # every FCR block covers 4 hours

        for day in results_all:
            P_DA      = np.asarray(day["P_DA"])
            P_grid    = np.asarray(day["P_grid"])
            h_sell_RT = np.asarray(day["h_sell_RT"])
            imbalance = np.asarray(day["imbalance"])        # P_grid − P_DA
            wind_real = np.asarray(day["wind_real"])
            P_curt_RT = np.asarray(day.get("P_curt_RT", np.zeros_like(P_DA)))
            P_elec_RT = np.asarray(day.get("P_elec_RT", np.zeros_like(P_DA)))
            P_buy_RT  = np.asarray(day.get("P_buy_RT",  np.zeros_like(P_DA)))

            price_DA  = np.asarray(day["price_DA"])    # EUR/MWh
            price_imb = np.asarray(day["price_imb"])   # EUR/MWh
            price_H2  = np.asarray(day["price_H2"])    # EUR/kg
            price_FCR = np.asarray(day["price_FCR"])   # EUR/kW per block
            DA_FCR    = np.asarray(day["DA_FCR"])       # kW per block

            # DA export revenue: P_DA [kW] × price [EUR/MWh] × dt [h] / 1000
            da_export_revenue_sim += float(np.sum(P_DA * price_DA * dt / 1000.0))

            # FCR capacity revenue: r_FCR [kW] × price [EUR/kW] × 4 h/block
            fcr_revenue_sim += float(np.sum(DA_FCR * price_FCR * dt_FCR))

            # Hydrogen revenue: h_sell [kg/slot] × price [EUR/kg] × dt [h]
            h2_revenue_sim  += float(np.sum(h_sell_RT * price_H2 * dt))
            total_h2_kg_sim += float(np.sum(h_sell_RT * dt))

            # Imbalance settlement (signed): positive = delivered more than bid
            imbalance_revenue_sim += float(
                np.sum(imbalance * price_imb * dt / 1000.0)
            )

            # Grid purchase cost (always a cost, so stored as positive number)
            grid_cost_sim += float(np.sum(P_buy_RT * price_DA * dt / 1000.0))

            # Energy aggregates
            total_wind_kwh_sim  += float(np.sum(wind_real  * dt))
            total_curt_kwh_sim  += float(np.sum(P_curt_RT  * dt))
            total_elec_kwh_sim  += float(np.sum(P_elec_RT  * dt))

        # Net revenue = revenues − grid purchase cost
        total_revenue_sim = (
            da_export_revenue_sim
            + fcr_revenue_sim
            + h2_revenue_sim
            + imbalance_revenue_sim
            - grid_cost_sim
        )

        # ── 2. Annualise ──────────────────────────────────────────────────────

        scale = 1.0 / self.cfg.simulation_years

        da_export_revenue_yr  = da_export_revenue_sim  * scale
        fcr_revenue_yr        = fcr_revenue_sim        * scale
        h2_revenue_yr         = h2_revenue_sim         * scale
        imbalance_revenue_yr  = imbalance_revenue_sim  * scale
        grid_cost_yr          = grid_cost_sim          * scale
        total_revenue_yr      = total_revenue_sim      * scale

        total_h2_kg_yr        = total_h2_kg_sim        * scale
        total_wind_kwh_yr     = total_wind_kwh_sim     * scale
        total_curt_kwh_yr     = total_curt_kwh_sim     * scale
        total_elec_kwh_yr     = total_elec_kwh_sim     * scale

        # ── 3. CAPEX & OPEX ───────────────────────────────────────────────────

        cfg = self.cfg

        capex_electrolyzer = cfg.electrolyzer_capacity_kw * cfg.capex_electrolyzer_eur_kw
        capex_wind         = cfg.wind_capacity_kw         * cfg.capex_wind_eur_kw
        capex_storage      = cfg.storage_capacity_kg      * cfg.capex_storage_eur_kg
        total_capex        = capex_electrolyzer + capex_wind + capex_storage

        opex_electrolyzer_yr = cfg.electrolyzer_capacity_kw * cfg.opex_electrolyzer_eur_kw_yr
        opex_wind_yr         = cfg.wind_capacity_kw         * cfg.opex_wind_eur_kw_yr
        total_opex_yr        = opex_electrolyzer_yr + opex_wind_yr

        # ── 4. LCOH ───────────────────────────────────────────────────────────
        # All-in: full system CAPEX + OPEX charged against annual H2 production

        crf              = _capital_recovery_factor(cfg.discount_rate, cfg.project_lifetime_yr)
        annualised_capex = total_capex * crf

        lcoh = (
            (annualised_capex + total_opex_yr) / total_h2_kg_yr
            if total_h2_kg_yr > 0
            else float("inf")
        )

        # ── 5. LCOE ───────────────────────────────────────────────────────────
        # Wind-only CAPEX + OPEX against gross wind generation

        annualised_capex_wind = (capex_wind + capex_storage) * crf
        lcoe = (
            (annualised_capex_wind + opex_wind_yr) / (total_wind_kwh_yr / 1000.0)
            if total_wind_kwh_yr > 0
            else float("inf")
        )

        # ── 6. NPV & IRR ──────────────────────────────────────────────────────
        # Year 0: −CAPEX
        # Year 1…N: (total revenue − OPEX) × (1 − tax_rate)

        ebitda_yr  = total_revenue_yr - total_opex_yr
        nopat_yr   = ebitda_yr * (1.0 - cfg.tax_rate)

        cash_flows = np.array(
            [-total_capex] + [nopat_yr] * cfg.project_lifetime_yr,
            dtype=float,
        )

        npv = _npv(cfg.discount_rate, cash_flows)
        irr = _irr(cash_flows)

        # ── 7. Physical / operational KPIs ────────────────────────────────────

        # Full-load hours: energy consumed / rated capacity
        flh_electrolyzer = (
            total_elec_kwh_yr / cfg.electrolyzer_capacity_kw
            if cfg.electrolyzer_capacity_kw > 0
            else 0.0
        )

        # Wind curtailment fraction
        curtailment_pct = (
            100.0 * total_curt_kwh_yr / total_wind_kwh_yr
            if total_wind_kwh_yr > 0
            else 0.0
        )

        # Simple payback (undiscounted)
        payback_yr = total_capex / nopat_yr if nopat_yr > 0 else float("inf")

        # ── 8. Store ──────────────────────────────────────────────────────────

        self._results = {
            # Revenue — simulated period
            "da_export_revenue_sim_eur":  da_export_revenue_sim,
            "fcr_revenue_sim_eur":        fcr_revenue_sim,
            "h2_revenue_sim_eur":         h2_revenue_sim,
            "imbalance_revenue_sim_eur":  imbalance_revenue_sim,
            "grid_cost_sim_eur":          grid_cost_sim,
            "total_revenue_sim_eur":      total_revenue_sim,

            # Revenue — annualised
            "da_export_revenue_yr_eur":   da_export_revenue_yr,
            "fcr_revenue_yr_eur":         fcr_revenue_yr,
            "h2_revenue_yr_eur":          h2_revenue_yr,
            "imbalance_revenue_yr_eur":   imbalance_revenue_yr,
            "grid_cost_yr_eur":           grid_cost_yr,
            "total_revenue_yr_eur":       total_revenue_yr,

            # Costs
            "capex_electrolyzer_eur":     capex_electrolyzer,
            "capex_wind_eur":             capex_wind,
            "capex_storage_eur":          capex_storage,
            "total_capex_eur":            total_capex,
            "total_opex_yr_eur":          total_opex_yr,
            "annualised_capex_eur":       annualised_capex,

            # KPIs
            "lcoh_eur_per_kg":            lcoh,
            "lcoe_eur_per_mwh":           lcoe,
            "npv_eur":                    npv,
            "irr":                        irr,
            "payback_yr":                 payback_yr,

            # Physical
            "total_h2_kg_yr":             total_h2_kg_yr,
            "total_wind_kwh_yr":          total_wind_kwh_yr,
            "total_curt_kwh_yr":          total_curt_kwh_yr,
            "total_elec_kwh_yr":          total_elec_kwh_yr,
            "flh_electrolyzer_hr":        flh_electrolyzer,
            "curtailment_pct":            curtailment_pct,
        }
        return self._results

    # ──────────────────────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        """Pretty-print results to stdout."""
        r   = self._results
        cfg = self.cfg
        if not r:
            print("No results yet — call compute() first.")
            return

        irr_str = f"{r['irr']*100:.1f}%" if r["irr"] is not None else "N/A"
        pb_str  = f"{r['payback_yr']:.1f}" if np.isfinite(r["payback_yr"]) else "∞"

        print("\n" + "=" * 62)
        print("  ECONOMIC ANALYSIS — WIND–HYDROGEN SYSTEM")
        print("=" * 62)

        # ── System overview ───────────────────────────────────────────────────
        print("\n── System overview ───────────────────────────────────────────")
        print(f"  Electrolyzer capacity         : {cfg.electrolyzer_capacity_kw/1000:>8.1f} MW")
        print(f"  Wind farm capacity (net)       : {cfg.wind_capacity_kw/1000:>8.1f} MW")
        print(f"  H2 storage capacity            : {cfg.storage_capacity_kg:>8.0f} kg")
        print(f"  Simulation period              : {cfg.simulation_years*365:.0f} days  "
              f"({cfg.simulation_years*100:.1f}% of year)")

        # ── Revenue: simulated period ─────────────────────────────────────────
        print("\n── Revenue breakdown (simulated period) ──────────────────────")
        print(f"  DA export revenue              : {r['da_export_revenue_sim_eur']:>12,.0f} EUR")
        print(f"  FCR capacity revenue           : {r['fcr_revenue_sim_eur']:>12,.0f} EUR")
        print(f"  Hydrogen sale revenue          : {r['h2_revenue_sim_eur']:>12,.0f} EUR")
        print(f"  Imbalance settlement           : {r['imbalance_revenue_sim_eur']:>12,.0f} EUR")
        print(f"  Grid purchase cost             : {-r['grid_cost_sim_eur']:>12,.0f} EUR")
        print(f"  ────────────────────────────────────────────────────────────")
        print(f"  NET TOTAL (simulated period)   : {r['total_revenue_sim_eur']:>12,.0f} EUR")

        # ── Revenue: annualised ───────────────────────────────────────────────
        print("\n── Revenue breakdown (annualised to full year) ───────────────")
        print(f"  DA export revenue              : {r['da_export_revenue_yr_eur']:>12,.0f} EUR/yr")
        print(f"  FCR capacity revenue           : {r['fcr_revenue_yr_eur']:>12,.0f} EUR/yr")
        print(f"  Hydrogen sale revenue          : {r['h2_revenue_yr_eur']:>12,.0f} EUR/yr")
        print(f"  Imbalance settlement           : {r['imbalance_revenue_yr_eur']:>12,.0f} EUR/yr")
        print(f"  Grid purchase cost             : {-r['grid_cost_yr_eur']:>12,.0f} EUR/yr")
        print(f"  ────────────────────────────────────────────────────────────")
        print(f"  NET TOTAL (annualised)         : {r['total_revenue_yr_eur']:>12,.0f} EUR/yr")

        # ── CAPEX & OPEX ──────────────────────────────────────────────────────
        print("\n── Capital & operating costs ─────────────────────────────────")
        print(f"  CAPEX — electrolyzer           : {r['capex_electrolyzer_eur']:>12,.0f} EUR")
        print(f"  CAPEX — wind farm              : {r['capex_wind_eur']:>12,.0f} EUR")
        print(f"  CAPEX — H2 storage             : {r['capex_storage_eur']:>12,.0f} EUR")
        print(f"  TOTAL CAPEX                    : {r['total_capex_eur']:>12,.0f} EUR")
        print(f"  Annualised CAPEX (CRF)         : {r['annualised_capex_eur']:>12,.0f} EUR/yr")
        print(f"  OPEX (annual)                  : {r['total_opex_yr_eur']:>12,.0f} EUR/yr")

        # ── Levelised costs ───────────────────────────────────────────────────
        print("\n── Levelised cost indicators ─────────────────────────────────")
        print(f"  LCOH                           : {r['lcoh_eur_per_kg']:>10.2f} EUR/kg H2")
        print(f"  LCOE (wind, gross)             : {r['lcoe_eur_per_mwh']:>10.2f} EUR/MWh")

        # ── Investment metrics ────────────────────────────────────────────────
        print("\n── Investment metrics ────────────────────────────────────────")
        print(f"  NPV                            : {r['npv_eur']:>12,.0f} EUR")
        print(f"  IRR                            : {irr_str:>12}")
        print(f"  Simple payback                 : {pb_str:>10} yr")

        # ── Physical / operational KPIs ───────────────────────────────────────
        print("\n── Physical & operational KPIs ───────────────────────────────")
        print(f"  H2 produced & sold (yr)        : {r['total_h2_kg_yr']:>12,.0f} kg/yr")
        print(f"  Wind energy available (yr)     : {r['total_wind_kwh_yr']/1e6:>10.2f} GWh/yr")
        print(f"  Wind energy curtailed (yr)     : {r['total_curt_kwh_yr']/1e6:>10.2f} GWh/yr  "
              f"({r['curtailment_pct']:.1f}%)")
        print(f"  Electrolyzer consumption (yr)  : {r['total_elec_kwh_yr']/1e6:>10.2f} GWh/yr")
        print(f"  Electrolyzer FLH               : {r['flh_electrolyzer_hr']:>10.0f} h/yr")
        print("=" * 62 + "\n")

    def to_dict(self) -> Dict:
        """Return a plain copy of all computed results."""
        return dict(self._results)


# ──────────────────────────────────────────────────────────────────────────────
# Financial helpers
# ──────────────────────────────────────────────────────────────────────────────

def _capital_recovery_factor(r: float, n: int) -> float:
    """CRF = r(1+r)^n / ((1+r)^n − 1)."""
    if r == 0:
        return 1.0 / n
    return r * (1 + r) ** n / ((1 + r) ** n - 1)


def _npv(rate: float, cash_flows: np.ndarray) -> float:
    """NPV where cash_flows[0] is year-0 outflow (negative CAPEX)."""
    t = np.arange(len(cash_flows))
    return float(np.sum(cash_flows / (1 + rate) ** t))


def _irr(cash_flows: np.ndarray, guess: float = 0.10) -> Optional[float]:
    """Newton-Raphson IRR.  Returns None if it fails to converge."""
    r = guess
    for _ in range(1_000):
        t    = np.arange(len(cash_flows))
        npv  = np.sum(cash_flows / (1 + r) ** t)
        dnpv = np.sum(-t * cash_flows / (1 + r) ** (t + 1))
        if abs(dnpv) < 1e-12:
            return None
        r_new = r - npv / dnpv
        if abs(r_new - r) < 1e-8:
            return float(r_new)
        r = max(r_new, -0.99)   # guard against division-by-zero
    return None