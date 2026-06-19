"""
seasonal_runner.py
------------------
Runs the wind–hydrogen optimisation for 4 seasonal 2-week windows and
saves all results to a single pickle file.

Usage:
    python seasonal_runner.py              # run all seasons
    python seasonal_runner.py --dry-run    # print config, no solve

Results file: results/seasonal_results.pkl
Load with:    import pickle; data = pickle.load(open("results/seasonal_results.pkl","rb"))
"""

import os, sys, pickle, argparse
import numpy as np

# ---------------------------------------------------------------------------
# Season definitions — adjust date ranges to match your ERA5 / SCADA data
# ---------------------------------------------------------------------------
SEASONS = {
    "winter": ("2026-01-07", "2026-01-21"),
    "spring": ("2026-04-07", "2026-04-21"),
    "summer": ("2026-07-07", "2026-07-21"),
    "autumn": ("2026-10-07", "2026-10-21"),
}

DAYS_PER_SEASON   = 14        # 2 weeks
TIMESTEPS_PER_DAY = 96        # 15-min slots
BLOCKS_PER_DAY    = TIMESTEPS_PER_DAY // 16   # 6 FCR blocks/day

# ---------------------------------------------------------------------------
# Shared optimisation settings  (mirror what is in main.py)
# ---------------------------------------------------------------------------
USE_STOCHASTIC = True
N_SCENARIOS    = 20
N_NEIGHBORS    = 50

# ---------------------------------------------------------------------------
# Imports that require the rest of the project to be on the path
# ---------------------------------------------------------------------------
from Data_preparation import DataPreparationCSV
from Scenario_gen    import compute_forecast_errors, generate_scenarios
from StochasticOptimizer import StochasticInputData, StochasticElectrolyzerModel
from Optimizer       import InputData, ElectrolyzerModel
import Hydrogen_production as hp
import numpy as np
from Piecewise import generate_piecewise   # adjust to your actual module name


# ---------------------------------------------------------------------------
# Build the electrolyzer piecewise once — shared across all seasons
# ---------------------------------------------------------------------------
def build_electrolyzer():
    A_cell     = 0.0625
    j_mA_cm2   = np.linspace(1, 400, 50)
    j          = j_mA_cm2 * 10
    I_range    = j * A_cell

    A_s, B_s, P_seg_min, P_seg_max, P_total, m_dot, P_stack = generate_piecewise(
        hp, T=80, I_range=I_range, A_cell=0.0625,
        p=60, p_out=300, N_stacks=40*30, N_cell=250, n_segments=3, plot=False,
    )

    P_total_curve, m_dot_curve, P_stack_curve = hp.electrolyzer_h2_prod_curve(
        T=80, I_range=I_range, A_cell=0.0625,
        p=60, p_out=300, N_stacks=40*30, N_cell=250,
    )

    P_max = float(np.max(P_stack_curve))
    P_min = 0.2 * P_max
    P_sb  = float(np.min(P_stack_curve))

    return dict(
        A_s=A_s, B_s=B_s,
        P_seg_min=P_seg_min, P_seg_max=P_seg_max,
        P_max=P_max, P_min=P_min, P_sb=P_sb,
        S=len(A_s),
    )


# ---------------------------------------------------------------------------
# Single-season runner  (mirrors the day loop in main.py)
# ---------------------------------------------------------------------------
def run_season(season_name: str, start_date: str, end_date: str,
               elec: dict, verbose: bool = True) -> dict:
    """
    Run DA+RT optimisation for every day of one seasonal window.

    Returns a dict with per-day results plus season-level aggregates.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  SEASON: {season_name.upper()}  ({start_date} → {end_date})")
        print(f"{'='*60}")

    # ---- load data for this season ----
    prep = DataPreparationCSV(
        start_date=start_date,
        end_date=end_date,
        el_price_file_name="DayAheadPrices.csv",
        wind_file_name="windspeeds.csv",
        fcr_file_name="FcrDK1.csv",
        imbalance_file_name="ImbalancePrice.csv",
        wind_source="era5",
        wtg_file_name="Vestas_V164_8MW.wtg",
    )
    data          = prep.data
    wind_df       = data["wind"]
    el_price_df   = data["el_price"]
    fcr_df        = data["fcr"]
    imbalance_df  = data["imbalance"]

    wind_real      = wind_df["wind_kW"].values
    price_real     = el_price_df["el_price[EUR/MWh]"].values
    price_H2       = np.full(len(wind_real), 4.5)
    price_FCR      = fcr_df["price_FCR"].values        # EUR/kW — already converted
    FCR_volume     = fcr_df["FCR_volume"].values
    imbalance_price = imbalance_df["ImbalancePriceEUR"].values

    # ---- forecast (lag-1-day) ----
    lag = 96
    price_errors, wind_errors, imb_errors, wind_forecasts = compute_forecast_errors(
        price_real, wind_real, imbalance_price, lag=lag,
    )
    wind_forecast      = np.roll(wind_real, lag);      wind_forecast[:lag]      = wind_real[:lag]
    price_forecast     = np.roll(price_real, lag);     price_forecast[:lag]     = price_real[:lag]
    imbalance_forecast = np.roll(imbalance_price, lag); imbalance_forecast[:lag] = imbalance_price[:lag]

    # ---- fixed parameters ----
    grid_cap      = 80_000.0
    h_min_daily   = 0.0
    h_sell_max    = 200_000.0
    storage_max   = 0.0
    N_down        = 6

    # ---- state that rolls across days ----
    storage_state = 0.0
    z_on_state    = 0
    P_prev        = 0.0

    num_days = min(DAYS_PER_SEASON, len(wind_real) // TIMESTEPS_PER_DAY)

    results_days = []
    DA_objectives = []
    RT_objectives = []

    for day in range(num_days):
        if verbose:
            print(f"  Day {day+1}/{num_days} ...", end=" ", flush=True)

        s  = day * TIMESTEPS_PER_DAY
        e  = s + TIMESTEPS_PER_DAY
        sb = day * BLOCKS_PER_DAY
        eb = sb + BLOCKS_PER_DAY

        price_FCR_day  = price_FCR[sb:eb]
        FCR_volume_day = FCR_volume[sb:eb]
        wind_fc_day    = wind_forecast[s:e]
        price_fc_day   = price_forecast[s:e]
        price_H2_day   = price_H2[s:e]
        imb_fc_day     = imbalance_forecast[s:e]

        # ---- DA solve ----
        if USE_STOCHASTIC:
            price_sc, wind_sc, imb_sc, probs = generate_scenarios(
                price_fc_day, wind_fc_day, imb_fc_day,
                price_errors, wind_errors, imb_errors, wind_forecasts,
                n_scenarios=N_SCENARIOS, n_neighbors=N_NEIGHBORS, mode="random",
            )
            input_DA = StochasticInputData(
                T=TIMESTEPS_PER_DAY, S_seg=elec["S"], I=len(price_FCR_day),
                wind_forecast=wind_fc_day,
                price_scenarios=price_sc, wind_scenarios=wind_sc,
                imb_scenarios=imb_sc, scenario_probs=probs,
                price_H2=price_H2_day, price_FCR=price_FCR_day,
                P_min=elec["P_min"], P_max=elec["P_max"], P_sb=elec["P_sb"],
                P_seg_min=elec["P_seg_min"], P_seg_max=elec["P_seg_max"],
                A_s=elec["A_s"], B_s=elec["B_s"],
                storage_max=storage_max, storage_initial=storage_state,
                h_min_daily=h_min_daily, h_sell_max=h_sell_max,
                grid_cap=grid_cap, N_down=N_down, FCR_volume=FCR_volume_day,
                z_on_prev=z_on_state, P_prev=P_prev,
            )
            DA_model = StochasticElectrolyzerModel(input_DA)
        else:
            input_DA = InputData(
                T=TIMESTEPS_PER_DAY, S=elec["S"], I=len(price_FCR_day),
                wind=None, wind_forecast=wind_fc_day,
                price_DA=price_fc_day, price_H2=price_H2_day,
                price_FCR=price_FCR_day, r_FCR_DA=None, imbalance_price=None,
                P_min=elec["P_min"], P_max=elec["P_max"], P_sb=elec["P_sb"],
                P_seg_min=elec["P_seg_min"], P_seg_max=elec["P_seg_max"],
                A_s=elec["A_s"], B_s=elec["B_s"],
                storage_max=storage_max, storage_initial=storage_state,
                h_min_daily=h_min_daily, h_sell_max=h_sell_max,
                grid_cap=grid_cap, N_down=N_down, FCR_volume=FCR_volume_day,
                z_on_prev=z_on_state, P_prev=P_prev, is_DA=True, DA_export=None,
            )
            DA_model = ElectrolyzerModel(input_DA)

        DA_model.run()

        P_DA       = np.array(DA_model.results.P_DA)
        h_sell_DA  = np.array(DA_model.results.h_sell)
        r_FCR_DA   = DA_model.results.r_FCR_block

        if USE_STOCHASTIC:
            P_Elec_DA = np.array([
                sum(DA_model.model.P_seg[0, t, s].value for s in DA_model.model.SEG)
                + DA_model.model.P_sb[0, t].value
                for t in DA_model.model.T
            ])
        else:
            P_Elec_DA = np.array([
                sum(DA_model.model.P_seg[t, s].value for s in DA_model.model.S)
                + DA_model.model.P_sb[t].value
                for t in DA_model.model.T
            ])

        DA_objectives.append(DA_model.results.objective_value)

        # ---- RT solve ----
        wind_rt_day   = wind_real[s:e]
        price_rt_day  = price_real[s:e]
        imbalance_day = imbalance_price[s:e]

        input_RT = InputData(
            T=TIMESTEPS_PER_DAY, S=elec["S"], I=len(price_FCR_day),
            wind=wind_rt_day, wind_forecast=wind_fc_day,
            price_DA=price_rt_day, price_H2=price_H2_day,
            price_FCR=price_FCR_day, r_FCR_DA=r_FCR_DA,
            imbalance_price=imbalance_day,
            P_min=elec["P_min"], P_max=elec["P_max"], P_sb=elec["P_sb"],
            P_seg_min=elec["P_seg_min"], P_seg_max=elec["P_seg_max"],
            A_s=elec["A_s"], B_s=elec["B_s"],
            storage_max=storage_max, storage_initial=storage_state,
            h_min_daily=h_min_daily, h_sell_max=h_sell_max,
            grid_cap=grid_cap, N_down=N_down, FCR_volume=FCR_volume_day,
            z_on_prev=z_on_state, P_prev=P_prev,
            is_DA=False, DA_export=P_DA,
        )
        RT_model = ElectrolyzerModel(input_RT)
        RT_model.run()

        # ---- update rolling state ----
        storage_state = RT_model.results.storage[-1]
        z_on_state    = RT_model.results.z_on[-1]
        P_prev        = RT_model.results.P_elec[-1]
        RT_objectives.append(RT_model.results.objective_value)

        results_days.append({
            # schedules
            "P_DA":        P_DA,
            "P_grid":      np.array(RT_model.results.P_grid),
            "P_elec_DA":   P_Elec_DA,
            "P_elec_RT":   np.array(RT_model.results.P_elec),
            "P_curt_DA":   np.array(DA_model.results.P_curt),
            "P_curt_RT":   np.array(RT_model.results.P_curt),
            "P_buy_RT":    np.array(RT_model.results.P_buy),
            # hydrogen
            "h_sell_DA":   h_sell_DA,
            "h_sell_RT":   np.array(RT_model.results.h_sell),
            "storage":     np.array(RT_model.results.storage),
            # FCR
            "FCR_blocks":  RT_model.results.r_FCR_block,
            "DA_FCR":      r_FCR_DA,
            # imbalance
            "imbalance":   np.array(RT_model.results.P_grid) - P_DA,
            # state
            "z_on_RT":     np.array(RT_model.results.z_on),
            "z_off_RT":    np.array(RT_model.results.z_off),
            # prices / exogenous
            "wind_real":   wind_rt_day,
            "wind_forecast": wind_fc_day,
            "price_DA":    price_rt_day,
            "price_imb":   imbalance_day,
            "price_H2":    price_H2_day,
            "price_FCR":   price_FCR_day,
        })

        if verbose:
            fcr_committed = int(np.any(r_FCR_DA > 0))
            print(f"DA obj={DA_model.results.objective_value:,.0f}  "
                  f"RT obj={RT_model.results.objective_value:,.0f}  "
                  f"FCR={'yes' if fcr_committed else 'no'}")

    # ---- season-level aggregates ----
    dt = 0.25
    all_h_sell   = np.concatenate([d["h_sell_RT"] for d in results_days])
    all_P_grid   = np.concatenate([d["P_grid"]    for d in results_days])
    all_P_DA     = np.concatenate([d["P_DA"]      for d in results_days])
    all_P_buy    = np.concatenate([d["P_buy_RT"]  for d in results_days])
    all_imb      = np.concatenate([d["imbalance"] for d in results_days])
    all_imb_price= np.concatenate([d["price_imb"] for d in results_days])
    all_price_DA = np.concatenate([d["price_DA"]  for d in results_days])
    all_price_H2 = np.concatenate([d["price_H2"]  for d in results_days])
    all_price_FCR= np.concatenate([d["price_FCR"] for d in results_days])
    all_FCR      = np.concatenate([d["FCR_blocks"] for d in results_days])
    all_wind     = np.concatenate([d["wind_real"]  for d in results_days])
    all_P_curt   = np.concatenate([d["P_curt_RT"]  for d in results_days])

    rev_DA   = float(np.sum(all_P_DA  * all_price_DA  * dt / 1000))
    rev_imb  = float(np.sum(all_imb   * all_imb_price * dt / 1000))
    rev_H2   = float(np.sum(all_h_sell * all_price_H2 * dt))
    cost_buy = float(np.sum(all_P_buy  * all_price_DA  * dt / 1000))
    rev_FCR  = float(np.sum(all_FCR    * all_price_FCR * 4))   # 4h blocks

    season_result = {
        # ---- metadata ----
        "season":     season_name,
        "start_date": start_date,
        "end_date":   end_date,
        "num_days":   num_days,
        # ---- per-day granularity ----
        "days":          results_days,
        "DA_objectives": DA_objectives,
        "RT_objectives": RT_objectives,
        # ---- season totals ----
        "revenue": {
            "DA_export":     rev_DA,
            "imbalance":     rev_imb,
            "hydrogen":      rev_H2,
            "grid_purchase": cost_buy,
            "FCR":           rev_FCR,
            "net":           rev_DA + rev_imb + rev_H2 - cost_buy + rev_FCR,
        },
        "H2_kg_sold":       float(np.sum(all_h_sell) * dt),
        "wind_GWh":         float(np.sum(all_wind)   * dt / 1e6),
        "curtailed_GWh":    float(np.sum(all_P_curt) * dt / 1e6),
        "FCR_blocks_committed": int(np.sum(all_FCR > 0)),
    }

    if verbose:
        r = season_result["revenue"]
        print(f"\n  ── {season_name} totals ──────────────────────────")
        print(f"  Net revenue : {r['net']:>12,.0f} EUR")
        print(f"  H2 sold     : {season_result['H2_kg_sold']:>12,.0f} kg")
        print(f"  Wind avail  : {season_result['wind_GWh']:>12.2f} GWh")
        print(f"  FCR blocks  : {season_result['FCR_blocks_committed']:>6} committed")

    return season_result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main(dry_run: bool = False):
    os.makedirs("results", exist_ok=True)
    out_path = "results/seasonal_results.pkl"

    print("Building electrolyzer model ...")
    elec = build_electrolyzer()
    print(f"  P_max={elec['P_max']:.0f} kW  P_min={elec['P_min']:.0f} kW  "
          f"P_sb={elec['P_sb']:.1f} kW  S={elec['S']} segments")

    if dry_run:
        print("\n[dry-run] Season schedule:")
        for name, (s, e) in SEASONS.items():
            print(f"  {name:8s}: {s} → {e}  ({DAYS_PER_SEASON} days)")
        return

    all_results = {}

    for season_name, (start, end) in SEASONS.items():
        result = run_season(season_name, start, end, elec, verbose=True)
        all_results[season_name] = result

        # save after each season so a crash doesn't lose everything
        with open(out_path, "wb") as f:
            pickle.dump(all_results, f)
        print(f"  [saved → {out_path}]")

    # ---- print combined summary ----
    print(f"\n{'='*60}")
    print("  ALL SEASONS COMBINED")
    print(f"{'='*60}")

    for key in ["DA_export", "imbalance", "hydrogen", "grid_purchase", "FCR", "net"]:
        total = sum(r["revenue"][key] for r in all_results.values())
        print(f"  {key:16s}: {total:>14,.0f} EUR")

    total_H2   = sum(r["H2_kg_sold"]    for r in all_results.values())
    total_wind = sum(r["wind_GWh"]      for r in all_results.values())
    total_curt = sum(r["curtailed_GWh"] for r in all_results.values())
    print(f"  {'H2 sold':16s}: {total_H2:>14,.0f} kg")
    print(f"  {'Wind available':16s}: {total_wind:>14.2f} GWh")
    print(f"  {'Curtailed':16s}: {total_curt:>14.2f} GWh  "
          f"({100*total_curt/total_wind:.1f}%)")

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)