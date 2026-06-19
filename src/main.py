from kiwisolver import Constraint

from Piecewise import generate_piecewise
import Hydrogen_production as hp
import numpy as np
import pandas as pd
from Data_preparation import DataPreparationCSV
from Optimizer import InputData, ElectrolyzerModel
from Economics import EconomicsAnalysis, EconomicsConfig
from pyomo.environ import value
#from Sensitivity import SizingStudy, SizingConfig
import matplotlib.pyplot as plt
import os
os.makedirs("results", exist_ok=True)
import pickle

#import matplotlib.gridspec as gridspec
#from matplotlib.colors import TwoSlopeNorm




USE_STOCHASTIC = False
N_SCENARIOS    = 20
N_NEIGHBORS    = 50

from Scenario_gen import compute_forecast_errors, generate_scenarios_cached
from StochasticOptimizer import StochasticInputData, StochasticElectrolyzerModel



# # --- Use SCADA data (default) ---
# prep = DataPreparationCSV(
#     start_date="2025-03-05",
#     end_date="2026-03-05",
#     el_price_file_name="DayAheadPrices.csv",
#     wind_file_name="WindProd5MinRealtime.csv",
#     fcr_file_name="FcrDK1.csv",
#     imbalance_file_name="ImbalancePrice.csv",
#     wind_source="scada",               # ← switch here
# )

# --- Use ERA5 + power curve instead ---
prep = DataPreparationCSV(
    start_date="2025-10-07",
    end_date="2025-12-30",
    el_price_file_name="DayAheadPrices.csv",
    wind_file_name="windspeeds.csv",   # ← ERA5 file
    fcr_file_name="FcrDK1.csv",
    imbalance_file_name="ImbalancePrice.csv",
    wind_source="era5",                # ← switch here
    wtg_file_name="Vestas_V164_8MW.wtg",
)   

data = prep.data

wind_df = data['wind']
el_price_df = data['el_price']
fcr_df = data['fcr']
imbalance_df = data['imbalance']
imbalance_price = imbalance_df['ImbalancePriceEUR'].values
wind_capacity_kw = wind_df['wind_kW'].max() 
print("Data prepared.")

# Count negative prices
negative_prices_count = (el_price_df['el_price[EUR/MWh]'] < 0).sum()
print(f"Number of timesteps with negative prices: {negative_prices_count}")

# --- PIECEWISE APPROXIMATION --- #
print("Generating piecewise approximation...")
A_cell = 0.0625 # m²
j_mA_cm2 = np.linspace(1, 400, 50)
j = j_mA_cm2 * 10          # A/m²
I_range = j * A_cell

A_s, B_s, P_seg_min, P_seg_max, P_total, m_dot, P_stack = generate_piecewise(
    hp,
    T=80,
    I_range=I_range,
    A_cell=0.0625,
    p=60,
    p_out=300,
    N_stacks=40*30,
    N_cell=250,
    n_segments=3,
    plot=True  # Set to True to see plot
)

print("Piecewise approximation generated.")
print("A_s:", A_s)
print("Typical efficiency (kg/kWh):", A_s)

# --- OPTIMIZATION SETUP --- #
print("Setting up optimization...")

# Time parameters
#H = 96  # e.g. 1 day (96 × 15min)
S = len(A_s)      # Number of segments
I = len(fcr_df)   # Number of FCR blocks (4-hour intervals)
# Electrolyzer parameters
P_total, m_dot, P_stack = hp.electrolyzer_h2_prod_curve(
    T=80,
    I_range=I_range,
    A_cell=0.0625,
    p=60,
    p_out=300,
    N_stacks=40*30,
    N_cell=250
)

P_max = max(P_stack) # Maximum power [kW] (adjust based on system)
P_min = 0.15*P_max     # Minimum power [kW] #CHANGE?
P_sb = P_max*0.1    # Standby power [kW] #CHANGE###??



# Data
wind_real = wind_df['wind_kW'].values  # Wind power [kW]
price_real = el_price_df['el_price[EUR/MWh]'].values  # Day-ahead price [EUR/MWh]
price_H2 = np.full(len(wind_real), 4.5) #EUR/kg
#price_H2 = np.full(H, 5.5)  # Hydrogen price [EUR/kg], constant for now, from European Hydrogen Observatory
price_FCR = fcr_df['price_FCR'].values  # FCR price [EUR/kW]
FCR_volume = fcr_df['FCR_volume'].values # FCR volume [kW], using "up" as example
grid_cap = P_max  # Max grid power [kW], adjust as needed

#Add "Forecast" that is just the day before repeated
lag = 96  # 1 day

# --- Compute historical forecast errors (once, before day loop) ---
price_errors, wind_errors, imb_errors, wind_forecasts = compute_forecast_errors(
    price_real, wind_real, imbalance_price, lag=96
)

 #FCR is daily blocks, not per timestep



# For the first day, fill with actual values (or zeros)
wind_forecast = np.roll(wind_real, lag)
price_forecast = np.roll(price_real, lag)
imbalance_forecast = np.roll(imbalance_price, lag)

wind_forecast[:lag] = wind_real[:lag]
price_forecast[:lag] = price_real[:lag]
imbalance_forecast[:lag] = imbalance_price[:lag]

# print("NaNs in FCR price:", np.isnan(price_FCR).sum())
# print("NaNs in FCR volume:", np.isnan(FCR_volume).sum())

#r_FCR_block = 0  ####TEMPORARILY WHILE TESTING #####
# Storage parameters
storage_initial = 0.0
#eff              = float(np.mean(A_s))
#max_prod_per_slot = P_max * eff               # peak kg/slot ≈ 696
h_min_daily  = 0#P_max * eff * 24*0.55            # 50% of max daily = 34.0 t
h_sell_max   = 2000000#h_min_daily /24*1.1                 # flat delivery rate = 354 kg/slot
storage_max  = 0#2 * h_min_daily                   # 2 days buffer = 68.0 t


N_down = 6             # Downtime in 15min intervals, adjust as needed

t0 = 0

storage_state = storage_initial
z_on_state = 0
P_prev = 0

TIMESTEPS_PER_DAY = 96
num_days =  30#len(wind_real) // TIMESTEPS_PER_DAY

blocks_per_day = TIMESTEPS_PER_DAY // 16


results_all = []
DA_objectives = []
RT_objectives = []

for day in range(num_days):


    print(f"Processing day {day + 1} of {num_days}...")


    start = day * TIMESTEPS_PER_DAY
    end   = (day + 1) * TIMESTEPS_PER_DAY

    start_block = day * blocks_per_day
    end_block   = (day + 1) * blocks_per_day

    price_FCR_day = price_FCR[start_block:end_block]      # length 6
    FCR_volume_day = FCR_volume[start_block:end_block]    # length 6

    wind_fc_day   = wind_forecast[start:end]
    price_fc_day  = price_forecast[start:end]
    price_H2_day  = price_H2[start:end]
    imbalance_fc_day = imbalance_forecast[start:end]
    # price_FCR_day = price_FCR[day:day+1]
    # FCR_volume_day = FCR_volume[day:day+1] #
    # print(f"\n=== DAY {day+1} INITIAL STATE ===")
    # print("storage_state:", storage_state)
    # print("z_on_state:", z_on_state)
    # print("P_prev:", P_prev)

    # --- 4. Build model ---
    if USE_STOCHASTIC:
        price_sc, wind_sc, imb_sc, probs = generate_scenarios_cached(
            price_fc_day, wind_fc_day, imbalance_fc_day,
            price_errors, wind_errors, imb_errors, wind_forecasts,
            n_scenarios=N_SCENARIOS, n_neighbors=N_NEIGHBORS,
            mode="random",
            day=day,
            run_id="spring_20",
        )
        input_DA = StochasticInputData(
            T=TIMESTEPS_PER_DAY, S_seg=S, I=len(price_FCR_day),
            wind_forecast=wind_fc_day,
            price_scenarios=price_sc,
            wind_scenarios=wind_sc,
            imb_scenarios=imb_sc,
            scenario_probs=probs,
            price_H2=price_H2_day,
            price_FCR=price_FCR_day,
            P_min=P_min, P_max=P_max, P_sb=P_sb,
            P_seg_min=P_seg_min, P_seg_max=P_seg_max,
            A_s=A_s, B_s=B_s,
            storage_max=storage_max, storage_initial=storage_state,
            h_min_daily=h_min_daily, h_sell_max=h_sell_max,
            grid_cap=grid_cap, N_down=N_down,
            FCR_volume=FCR_volume_day,
            z_on_prev=z_on_state, P_prev=P_prev
        )
        DA_model = StochasticElectrolyzerModel(input_DA)
    else:
        input_DA = InputData(
            T=TIMESTEPS_PER_DAY,
            S=S,
            I=len(price_FCR_day),  # number of FCR blocks
            wind=None,
            wind_forecast=wind_fc_day,
            price_DA=price_fc_day,
            price_H2=price_H2_day,
            price_FCR=price_FCR_day,
            r_FCR_DA = None,
            imbalance_price=None,
            P_min=P_min,
            P_max=P_max,
            P_sb=P_sb,
            P_seg_min=P_seg_min,
            P_seg_max=P_seg_max,
            A_s=A_s,
            B_s=B_s,
            storage_max=storage_max,
            storage_initial=storage_state,
            h_min_daily=h_min_daily,
            h_sell_max=h_sell_max,
            grid_cap=grid_cap,
            N_down=N_down,
            FCR_volume=FCR_volume_day,
            z_on_prev=z_on_state,
            P_prev=P_prev,
            is_DA=True,
            DA_export=None  # Not used in DA model
        )

        DA_model = ElectrolyzerModel(input_DA)
        # print("Running DA for day", day)
    # print("storage_state:", storage_state)
    # print("z_on_state:", z_on_state)
    # print("P_prev:", P_prev)
    DA_model.run()
    # print(f"\n--- Window starting at t0={t0} ---")
    # print(f"  storage_initial passed to model: {storage_state}")
    
    P_DA = np.array(DA_model.results.P_DA) # DA export schedule
    h_sell_DA = np.array(DA_model.results.h_sell)
    P_curt_DA = np.array(DA_model.results.P_curt)
    #P_buy_DA = np.array(DA_model.results.P_buy)
    DA_objectives.append(DA_model.results.objective_value)
    r_FCR_DA = DA_model.results.r_FCR_block
    
    # Handle both deterministic and stochastic models
    if USE_STOCHASTIC:
        # For stochastic model, average across scenarios, show first scenario
        P_Elec_DA = np.array([
            sum(DA_model.model.P_seg[0, t, s].value for s in DA_model.model.SEG)
            + DA_model.model.P_sb[0, t].value
            for t in DA_model.model.T
        ])
    else:
        # For deterministic model
        P_Elec_DA = np.array([
            sum(DA_model.model.P_seg[t, s].value for s in DA_model.model.S)
            + DA_model.model.P_sb[t].value
            for t in DA_model.model.T
        ])

    # if USE_STOCHASTIC:
    #     print(f"\n=== DAY {day+1} FCR STAGE-1 RESULT ===")
    #     m = DA_model.model
    #     for i in m.I:
    #         zfcr = value(m.z_FCR[i])
    #         rfcr = value(m.r_FCR_block[i])
    #         price = value(m.price_FCR[i])
    #         vol   = DA_model.data.P_max_bid[i]
    #         print(f"  Block {i}: z_FCR={zfcr:.0f}  r_FCR={rfcr:.1f} kW  "
    #             f"price={price:.4f}  max_bid={vol:.1f} kW")
            
    #     print(f"\n=== DAY {day+1} z_on PER SCENARIO (first t of each block) ===")
    #     for i in m.I:
    #         t0 = i * 16
    #         zon_by_sc = [value(m.z_on[sc, t0]) for sc in m.SC]
    #         min_zon   = min(zon_by_sc)
    #         n_off     = sum(1 for z in zon_by_sc if z < 0.5)
    #         print(f"  Block {i} t={t0}: min_z_on={min_zon:.0f}  "
    #             f"scenarios with z_on=0: {n_off}/{DA_model.data.N_sc}")

    #     # Also check all 16 timesteps for the first block
    #     print("\n  Block 0 full 16 timesteps (scenario 0 only):")
    #     for t_off in range(16):
    #         t = t_off
    #         if t < DA_model.data.T:
    #             print(f"    t={t}: z_on[0,{t}]={value(m.z_on[0, t]):.0f}"
    #                 f"  z_sb={value(m.z_sb[0, t]):.0f}"
    #                 f"  wind={value(m.wind_sc[0, t]):.1f} kW"
    #                 f"  price={value(m.price_sc[0, t]):.2f}")
                
    #         print(f"\n=== DAY {day+1} HEADROOM CHECK (worst-case scenario) ===")
    #         d = DA_model.data
    #         for i in m.I:
    #             # For each scenario, compute headroom = P_max - P_elec - P_sb
    #             headrooms = []
    #             for sc in m.SC:
    #                 for t_off in range(16):
    #                     t = i * 16 + t_off
    #                     if t >= d.T:
    #                         continue
    #                     p_elec = sum(value(m.P_seg[sc, t, s]) for s in m.SEG)
    #                     p_sb   = value(m.P_sb[sc, t])
    #                     z_on   = value(m.z_on[sc, t])
    #                     headroom = d.P_max * (z_on + value(m.z_sb[sc, t])) - p_elec - p_sb
    #                     headrooms.append(headroom)
    #             print(f"  Block {i}: min_headroom={min(headrooms):.1f} kW  "
    #                 f"mean_headroom={sum(headrooms)/len(headrooms):.1f} kW")

    #         # Check min_load slack: P_elec - r_FCR >= P_min * z_on
    #         # With r_FCR=0, slack = P_elec - P_min*z_on
    #         print("\n  Min-load slack with r_FCR=0 (block 1, all scenarios):")
    #         for sc in m.SC:
    #             slacks = []
    #             for t_off in range(16):
    #                 t = 16 + t_off
    #                 if t >= d.T: continue
    #                 p_elec = sum(value(m.P_seg[sc, t, s]) for s in m.SEG)
    #                 z_on   = value(m.z_on[sc, t])
    #                 slack  = p_elec - d.P_min * z_on
    #                 slacks.append(slack)
    #             print(f"    sc={sc}: min_slack={min(slacks):.1f} kW")

    #     # Run AFTER the normal DA solve to test feasibility in isolation.

    #     m2 = DA_model.model

    #     # Fix block 1 to participate
    #     m2.z_FCR[1].fix(1)
    #     m2.r_FCR_block[1].fix(1000)   # minimum bid, kW

    #     from pyomo.opt import SolverFactory
    #     solver2 = SolverFactory('gurobi')
    #     solver2.options['DualReductions'] = 0
    #     res2 = solver2.solve(m2, tee=False)
    #     print(f"\n=== FORCED FCR TEST (block 1, r=1000 kW) ===")
    #     print(f"  Termination: {res2.solver.termination_condition}")
    #     if str(res2.solver.termination_condition) == 'optimal':
    #         print(f"  Objective: {value(m2.obj):.2f}")
    #         print("  → FCR IS FEASIBLE. The issue is economic, not structural.")
    #     else:
    #         print("  → FCR IS INFEASIBLE. There is a hard constraint conflict.")
    #         from pyomo.util.infeasible import log_infeasible_constraints
    #         import logging
    #         logging.getLogger('pyomo.core').setLevel(logging.INFO)
    #         log_infeasible_constraints(m2, log_expression=True, tol=1e-4)

    #     # Unfix so the rest of the loop continues normally
    #     m2.z_FCR[1].unfix()
    #     m2.r_FCR_block[1].unfix()

    #     # This prints the one number that explains the gap between
    # # stochastic and deterministic FCR participation.
    #     print(f"\n=== STOCHASTIC vs DETERMINISTIC GAP ===")
    #     d = DA_model.data
    #     for i in m.I:
    #         # What r_FCR does max_cap_rule allow, per scenario?
    #         # r_FCR <= P_max*(z_on+z_sb) - P_elec - P_sb   for ALL sc, ALL t
    #         # The binding scenario is the one with minimum headroom.
    #         min_hw = float('inf')
    #         binding_sc, binding_t = -1, -1
    #         for sc in m.SC:
    #             for t_off in range(16):
    #                 t = i * 16 + t_off
    #                 if t >= d.T: continue
    #                 p_elec = sum(value(m.P_seg[sc, t, s]) for s in m.SEG)
    #                 p_sb   = value(m.P_sb[sc, t])
    #                 cap    = d.P_max * (value(m.z_on[sc,t]) + value(m.z_sb[sc,t]))
    #                 hw     = cap - p_elec - p_sb
    #                 if hw < min_hw:
    #                     min_hw = hw
    #                     binding_sc, binding_t = sc, t
    #         print(f"  Block {i}: max feasible r_FCR (from headroom) = {min_hw:.1f} kW "
    #             f"  ← binding sc={binding_sc}, t={binding_t}")
    #         # Compare: what would the deterministic model see?
    #         # (Uses wind_forecast, single 'scenario')
    #         t0 = i * 16
    #         print(f"           FCR min_bid = {d.P_min_bid} kW  "
    #             f"  z_FCR_min_bid feasible: {'YES' if min_hw >= d.P_min_bid else 'NO'}")
        # print("\n=== DAY 2 DA SUMMARY ===")
    # print("P_DA first 10:", P_DA[:10])
    # print("P_elec_DA first 10:", P_Elec_DA[:10])
    # print("h_sell_DA first 10:", h_sell_DA[:10])
    # print("storage_end_DA:", DA_model.results.storage[-1])
    # print("z_on_end_DA:", DA_model.results.z_on[-1])

    # -----------------------------
    # 3. Slice REAL DATA for RT run
    # -----------------------------
    wind_rt_day   = wind_real[start:end]
    price_rt_day  = price_real[start:end]
    imbalance_day = imbalance_price[start:end]
    price_H2_RT   = price_H2[start:end]
    #price_H2_RT = np.full(H, 5.5)

    input_RT = InputData(
        T=TIMESTEPS_PER_DAY,
        S=S,
        I=len(price_FCR_day),
        wind=wind_rt_day,             # actual wind
        wind_forecast=wind_fc_day,         # same forecast as DA
        price_DA=price_rt_day,        # actual DA price
        price_H2=price_H2_RT,
        price_FCR=price_FCR_day,
        r_FCR_DA=r_FCR_DA,          # use DA FCR schedule
        imbalance_price=imbalance_day,
        P_min=P_min,
        P_max=P_max,
        P_sb=P_sb,
        P_seg_min=P_seg_min,
        P_seg_max=P_seg_max,
        A_s=A_s,
        B_s=B_s,
        storage_max=storage_max,
        storage_initial=storage_state,
        h_min_daily=h_min_daily,
        h_sell_max=h_sell_max,
        grid_cap=grid_cap,
        N_down=N_down,
        FCR_volume=FCR_volume_day,
        z_on_prev=z_on_state,
        P_prev=P_prev,
        is_DA=False,
        DA_export=P_DA             # fixed DA bid
    )


    RT_model = ElectrolyzerModel(input_RT)
    print("\n=== DAY %d RT INPUT STATE ===" % (day + 1))
    print("storage_initial:", storage_state)
    print("z_on_prev:", z_on_state)
    print("P_prev:", P_prev)
    print("r_FCR_DA:", r_FCR_DA)
    print("wind_rt_day first 10:", wind_rt_day[:10])
    print("DA_export first 10:", P_DA[:10])
    print("imbalance first 10:", imbalance_day[:10])
    RT_model.run()

    h_sell_RT = np.array(RT_model.results.h_sell)
    P_grid    = np.array(RT_model.results.P_grid)
    #P_buy_RT  = np.array(RT_model.results.P_buy)

    RT_objectives.append(RT_model.results.objective_value)
    P_curt_RT = np.array(RT_model.results.P_curt)
    #Update states for next day
    storage_state = RT_model.results.storage[-1]
    z_on_state = RT_model.results.z_on[-1]
    P_prev = RT_model.results.P_elec[-1]

    print(f"\nDAY {day+1}")
    print("  FCR blocks this day:", price_FCR_day)
    print("  len(price_FCR_day):", len(price_FCR_day))
    print("  I passed to model:", len(price_FCR_day))
    print("_-----------------------------")
    print(f"\nDAY {day+1} — DA FCR results")
    print("  r_FCR_DA:", r_FCR_DA)
    print("  price_FCR_day:", price_FCR_day)
    print("  FCR_volume_day:", FCR_volume_day)
    print("  r_FCR_DA:", r_FCR_DA)
    print("  P_min:", P_min, "P_max:", P_max)
    print("  first few wind_rt_day:", wind_rt_day[:10])
    print("  storage_state at start:", storage_state)

    #Save results
    day_date = pd.to_datetime(prep.start_date) + pd.Timedelta(days=day)
    results_all.append({
        "date":         day_date.strftime("%Y-%m-%d"),
        "P_DA": P_DA,
        "P_grid": P_grid,
        "h_sell_DA": h_sell_DA,
        "h_sell_RT": h_sell_RT,
        "h_RT":     np.array(RT_model.results.h),
        "P_curt_DA": P_curt_DA,
        #"P_buy_RT": P_buy_RT,
        "imbalance": np.array(P_grid) - np.array(P_DA),
        "Total profit": DA_model.results.objective_value + RT_model.results.objective_value - sum(h_sell_DA * price_H2_day),
        "P_curt_RT": np.array(RT_model.results.P_curt),
        "P_elec_RT": np.array(RT_model.results.P_elec),
        "P_elec_DA": P_Elec_DA,
        "imbalance": np.array(P_grid) - np.array(P_DA),
        "storage_end": storage_state,
        "storage": np.array(RT_model.results.storage),
        "FCR_blocks": RT_model.results.r_FCR_block,   # array shape (I,)
        "DA_FCR": DA_model.results.r_FCR_block,       # array shape (I,)
        "NameDA_FCR": r_FCR_DA,
        "z_on_RT": np.array(RT_model.results.z_on),
        "z_off_RT": np.array(RT_model.results.z_off),
        "wind_real": wind_rt_day,
        "wind_forecast": wind_fc_day,
        "price_imb":   imbalance_day,       # EUR/MWh
        "price_DA":    price_rt_day,        # EUR/MWh
        "price_H2":    price_H2_RT,         # EUR/kg
        "price_FCR":   price_FCR_day ,       #EUR/kW
    })

 

cfg = EconomicsConfig(
    # ── system sizes: all derived from data or model, never hard-coded ────────
    electrolyzer_capacity_kw    = P_max,                  # from piecewise / hp
    wind_capacity_kw            = wind_capacity_kw,       # peak net wind [kW]
    storage_capacity_kg         = storage_max,            # from main setup
 
    # ── CAPEX / OPEX / financial: edit once, here only ────────────────────────
    capex_electrolyzer_eur_kw   = 1_000.0,   # EUR/kW  — adjust to your estimate
    capex_wind_eur_kw           = 2_000.0,   # EUR/kW
    capex_storage_eur_kg        = 500.0,     # EUR/kg H2
 
    opex_electrolyzer_eur_kw_yr = 150.0,      # EUR/(kW·yr)
    opex_wind_eur_kw_yr         = 00.0,      # EUR/(kW·yr)
 
    discount_rate               = 0.05,
    project_lifetime_yr         = 20,
    simulation_years            = num_days / 365.0,   # fraction of year simulated
    tax_rate                    = 0.22,
)
 
econ = EconomicsAnalysis(cfg)
econ.compute(results_all)
econ.print_summary()
 
econ_results = econ.to_dict()   # plain dict — use for plotting / export
# --- Compute electrolyzer capacity factor from simulation results ---
dt = 0.25  # hours per timestep (15 min)
# Sum energy consumed by electrolyzer across all days (kWh)
elec_total_kwh = sum(np.sum(d.get("P_elec_RT", np.zeros(TIMESTEPS_PER_DAY))) * dt for d in results_all)
sim_days = len(results_all) if len(results_all) > 0 else num_days
# Scale to full year
elec_yr = elec_total_kwh * (365.0 / sim_days) if sim_days > 0 else 0.0
hours_per_year = 8760.0
rated_kw = P_max
capacity_factor = elec_yr / (rated_kw * hours_per_year) if (rated_kw * hours_per_year) > 0 else float('nan')
print(f"Electrolyzer capacity factor (annualised): {capacity_factor:.3%}")
econ_results["elec_consumed_kwh_yr"] = elec_yr
econ_results["capacity_factor"] = capacity_factor

from analysis import (
    electrolyzer_profitability,
    imbalance_decomposition,
    size_sensitivity,
    fcr_profitability,
)

# RQ1
df_prof, df_sens = electrolyzer_profitability(
    results_all, h2_prices=[3.0, 4.5, 6.0, 8.0],
    econ_cfg_base=cfg, P_el_kw=P_max, save_dir="figures/jan/")

df_imb = imbalance_decomposition(results_all, P_max=P_max, save_dir="figures/jan/")

# RQ2
df_fcr, breakeven = fcr_profitability(results_all, P_max=P_max, P_min=P_min,
                                       save_dir="figures/jan/")
 

###NY###

# =============================================================
# UNIT BASIS: kW, EUR/MWh, kg/h, h
# Revenue formula: kW * EUR/MWh * h / 1000 = EUR
#                  kg/h * EUR/kg * h        = EUR
#                  kW * EUR/kW/h * h        = EUR
# =============================================================
dt     = 0.25   # h per 15-min timestep
dt_FCR = 4.0    # h per FCR block

# -- Rebuild full arrays (safe to call multiple times) --
P_DA_full      = np.concatenate([d["P_DA"]      for d in results_all])
P_grid_full    = np.concatenate([d["P_grid"]    for d in results_all])
#P_buy_RT_full  = np.concatenate([d["P_buy_RT"]  for d in results_all])
P_curt_RT_full = np.concatenate([d["P_curt_RT"] for d in results_all])
P_elec_RT_full = np.concatenate([d["P_elec_RT"] for d in results_all])
P_elec_DA_full = np.concatenate([d["P_elec_DA"] for d in results_all])
h_sell_DA_full = np.concatenate([d["h_sell_DA"] for d in results_all])
h_sell_RT_full = np.concatenate([d["h_sell_RT"] for d in results_all])
storage_full   = np.concatenate([d["storage"]   for d in results_all])
z_on_full      = np.concatenate([d["z_on_RT"]   for d in results_all])
DA_FCR_full    = np.concatenate([d["DA_FCR"]    for d in results_all])
wind_real_full = np.concatenate([d["wind_real"] for d in results_all])
wind_fc_full   = wind_forecast[:len(P_DA_full)]

N = len(P_DA_full)
price_real_full = price_real[:N]
imb_price_full  = imbalance_price[:N]
price_H2_full   = price_H2[:N]
price_FCR_full  = price_FCR[:len(DA_FCR_full)]

# --- Revenue terms ---
rev_elec_DA  = np.sum(P_DA_full * price_real_full * dt / 1000)
rev_imb      = np.sum((P_grid_full - P_DA_full) * imb_price_full * dt / 1000)
rev_h2_RT    = np.sum(h_sell_RT_full * price_H2_full * dt)
#cost_buy     = np.sum(P_buy_RT_full  * price_real_full * dt / 1000)
rev_FCR      = np.sum(DA_FCR_full * price_FCR_full * dt_FCR)   # kW * EUR/kW/h * h = EUR

total = rev_elec_DA + rev_imb + rev_h2_RT + rev_FCR #- cost_buy

print("\n=== REALIZED REVENUE BREAKDOWN ===")
print(f"  DA electricity  : {rev_elec_DA:>12,.0f} EUR")
print(f"  Imbalance       : {rev_imb:>12,.0f} EUR")
print(f"  RT hydrogen     : {rev_h2_RT:>12,.0f} EUR")
#print(f"  Grid purchase   : {-cost_buy:>12,.0f} EUR")
print(f"  FCR             : {rev_FCR:>12,.0f} EUR")
print(f"  {'─'*30}")
print(f"  TOTAL           : {total:>12,.0f} EUR")

# --- H2 break-even ---
total_h2_kg = np.sum(h_sell_RT_full) * dt                              # kg
opp_cost    = np.sum(P_elec_RT_full * price_real_full * dt / 1000)    # EUR
if total_h2_kg > 0:
    breakeven = opp_cost / total_h2_kg
    margin    = price_H2[0] - breakeven
    print(f"\n  Total H₂ sold   : {total_h2_kg:>10,.0f} kg")
    print(f"  Elec opp. cost  : {opp_cost:>10,.0f} EUR")
    print(f"  Break-even H₂   : {breakeven:>10.2f} EUR/kg")
    print(f"  H₂ margin       : {margin:>+10.2f} EUR/kg "
          f"({'profitable' if margin > 0 else 'LOSS-MAKING'})")

# --- Operational summary ---
print(f"\n=== OPERATIONAL SUMMARY ===")
util = 100 * z_on_full.mean()
print(f"  Electrolyzer utilisation : {util:.1f} %")
print(f"  Total wind available     : {np.sum(wind_real_full)*dt/1e6:.3f} GWh")
print(f"  Total curtailed          : {np.sum(P_curt_RT_full)*dt/1e6:.3f} GWh")
print(f"  Curtailment rate         : {100*np.sum(P_curt_RT_full)/max(np.sum(wind_real_full),1):.1f} %")
print(f"  FCR blocks committed     : {int(np.sum(DA_FCR_full > 0))} / {len(DA_FCR_full)}")
if np.any(DA_FCR_full > 0):
    print(f"  Mean FCR (active blocks) : {DA_FCR_full[DA_FCR_full>0].mean():,.0f} kW")
print(f"  P_max electrolyzer       : {P_max:,.0f} kW")
print(f"  P_min electrolyzer       : {P_min:,.0f} kW")
print(f"  P_sb electrolyzer        : {P_sb:,.1f} kW")

# --- Wind balance check ---
lhs = wind_real_full #+ P_buy_RT_full
rhs = P_elec_RT_full + P_curt_RT_full + P_grid_full
residual = np.abs(lhs - rhs)
print(f"\n  Wind balance max residual : {residual.max():.2f} kW  "
      f"{'✓' if residual.max() < 1 else '!! CHECK MODEL'}")

# --- Plots (labels corrected to kW/MW) ---
fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

axes[0].plot(P_DA_full/1000, label="DA export (MW)", alpha=0.8)
axes[0].plot(P_grid_full/1000, label="RT export (MW)", alpha=0.8)
axes[0].plot(wind_real_full/1000, label="Wind (MW)", alpha=0.5, color='green')
axes[0].set_ylabel("Power [MW]")
axes[0].set_title("Power flows")
axes[0].legend()
axes[0].grid()

imbalance_full = P_grid_full - P_DA_full
forecast_error = wind_real_full - wind_fc_full
axes[1].plot(imbalance_full/1000, label="Imbalance (MW)", alpha=0.8)
axes[1].plot(forecast_error/1000, label="Wind forecast error (MW)", alpha=0.8)
axes[1].set_ylabel("Power [MW]")
axes[1].set_title("Imbalance and forecast error")
axes[1].legend()
axes[1].grid()

axes[2].plot(h_sell_RT_full, label="RT H₂ sold (kg/h)", alpha=0.8)
axes[2].plot(h_sell_DA_full, label="DA H₂ sold (kg/h)", alpha=0.6)
axes[2].set_ylabel("kg/h")
axes[2].set_xlabel("Timestep (15-min)")
axes[2].set_title("Hydrogen sales")
axes[2].legend()
axes[2].grid()

plt.tight_layout()
plt.show()

# --- Daily summary bar chart ---
days_idx = np.arange(num_days)
da_exp   = np.array([d["P_DA"].sum() for d in results_all]) * dt / 1000    # MWh
rt_exp   = np.array([d["P_grid"].sum() for d in results_all]) * dt / 1000  # MWh
h2_rt    = np.array([d["h_sell_RT"].sum() for d in results_all]) * dt      # kg
curt     = np.array([d["P_curt_RT"].sum() for d in results_all]) * dt / 1000  # MWh

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].bar(days_idx+1, da_exp, label='DA', alpha=0.8)
axes[0].bar(days_idx+1, rt_exp, bottom=da_exp, label='RT deviation', alpha=0.8)
axes[0].set_title("Daily grid export [MWh]")
axes[0].set_xlabel("Day")
axes[0].legend()
axes[0].grid(axis='y')

axes[1].bar(days_idx+1, h2_rt, color='orange')
axes[1].set_title("Daily RT H₂ sold [kg]")
axes[1].set_xlabel("Day")
axes[1].grid(axis='y')

axes[2].bar(days_idx+1, curt, color='red', alpha=0.7)
axes[2].set_title("Daily curtailment [MWh]")
axes[2].set_xlabel("Day")
axes[2].grid(axis='y')

plt.tight_layout()
plt.show()

#from plots_annual import plot_annual_results, plot_h2_sensitivity, override_h2_price

#plot_annual_results(results_all, econ_results, P_max, wind_capacity_kw,
#                    price_H2=price_H2[0], save_dir="figures/")

# H2 sensitivity — no new solver calls
#plot_h2_sensitivity(results_all, cfg, h2_prices=[3.0, 4.5, 6.0, 8.0],
#                    P_el_kw=P_max, save_dir="figures/")
def plot_imbalance_deep_dive(results_all, label, n_days=3, start_day=0):
    dt = 0.25

    days = results_all[start_day : start_day + n_days]

    P_DA      = np.concatenate([d["P_DA"]      for d in days])
    P_grid    = np.concatenate([d["P_grid"]    for d in days])
    P_elec_DA = np.concatenate([d["P_elec_DA"] for d in days])
    P_elec_RT = np.concatenate([d["P_elec_RT"] for d in days])
    #P_buy     = np.concatenate([d["P_buy_RT"]  for d in days])
    wind      = np.concatenate([d["wind_real"] for d in days])
    p_DA      = np.concatenate([d["price_DA"]  for d in days])
    p_imb     = np.concatenate([d["price_imb"] for d in days])

    deviation     = P_grid - P_DA
    price_diff    = p_imb - p_DA
    opportunistic = (deviation * price_diff) > 0

    T = len(P_DA)
    t = np.arange(T) * dt

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f"Imbalance deep-dive — {label}  (days {start_day}–{start_day+n_days-1})",
                 fontsize=13)

    # ── Panel 1: power flows + P_buy ─────────────────────────────────────
    ax = axes[0]
    ax.fill_between(t, 0, wind/1000, alpha=0.2, color="steelblue", label="Wind (real)")
    ax.plot(t, P_DA/1000,   color="navy",      lw=1.5, ls="--", label="P_DA committed")
    ax.plot(t, P_grid/1000, color="royalblue", lw=1.2,          label="P_grid realised")
    #ax.plot(t, P_buy/1000,  color="crimson",   lw=1.2, ls=":",  label="P_buy (grid import)")
    ax.set_ylabel("MW")
    ax.set_title("Grid export: DA commitment vs RT realisation")
    ax.legend(loc="upper right", fontsize=8)
    ax.axhline(0, color="gray", lw=0.5)

    # ── Panel 2: electrolyzer ─────────────────────────────────────────────
    ax = axes[1]
    ax.plot(t, P_elec_DA/1000, color="darkorange", lw=1.5, ls="--", label="Elec DA commitment")
    ax.plot(t, P_elec_RT/1000, color="orangered",  lw=1.2,          label="Elec RT realised")
    ax.set_ylabel("MW")
    ax.set_title("Electrolyzer: DA commitment vs RT realisation")
    ax.legend(loc="upper right", fontsize=8)

    # ── Panel 3: deviation ────────────────────────────────────────────────
    ax = axes[2]
    colors = np.where(opportunistic, "green", "red")
    ax.bar(t, deviation/1000, width=dt*0.9, color=colors, alpha=0.7)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("MW")
    ax.set_title("RT deviation (RT−DA): green = Profit, red = Penalty")
    # Correct: based on actual cash flow per slot
    imb_rev_per_slot = deviation * p_imb * dt / 1000   # EUR per slot
    colors = np.where(imb_rev_per_slot > 0, "green", "red")

    ax.bar(t, deviation/1000, width=dt*0.9, color=colors, alpha=0.7)

    # Update title to reflect what colours mean
    ax.set_title("RT deviation (RT−DA): green = cash gain, red = cash loss")

    # Cumulative uses the same per-slot revenue
    cumrev = np.cumsum(imb_rev_per_slot)
    ax2 = ax.twinx()
    ax2.plot(t, cumrev, color="purple", lw=1.5, label="Cumulative imb. revenue")
    ax2.set_ylabel("EUR (cumulative)", color="purple")
    ax2.tick_params(axis="y", labelcolor="purple")

    # ── Panel 4: prices + zero line ───────────────────────────────────────
    ax = axes[3]
    ax.axhline(0,   color="black", lw=1.2, ls="-",  zorder=3, label="0 EUR/MWh")  # zero line
    ax.axhline(0,   color="gray",  lw=0.5)                                          # subtle grid
    ax.plot(t, p_DA,  color="gray",    lw=1.5, label="DA price")
    ax.plot(t, p_imb, color="crimson", lw=1.2, label="Imbalance price")
    ax.fill_between(t, p_DA, p_imb,
                    where=(p_imb > p_DA), alpha=0.2, color="green",
                    label="Imb premium → up-regulation (system short)")
    ax.fill_between(t, p_DA, p_imb,
                    where=(p_imb < p_DA), alpha=0.2, color="red",
                    label="Imb discount → down-regulation (system long)")
    ax.fill_between(t, 0, p_imb,
                    where=(p_imb < 0), alpha=0.15, color="purple",
                    label="Negative imb price (curtailment pressure)")
    ax.set_ylabel("EUR/MWh")
    ax.set_xlabel("Hour of window")
    ax.set_title("Prices: DA vs imbalance  "
                 "(imb > DA = up-reg. signal,  imb < 0 = down reg. signal)")
    ax.legend(loc="upper right", fontsize=8)

    for ax in axes:
        for d in range(1, n_days):
            ax.axvline(d * 24, color="black", lw=0.8, ls=":")

    plt.tight_layout()
    plt.savefig(f"figures/imbalance_deepdive_{label.replace(' ','_')}.png",
                dpi=150, bbox_inches="tight")
    plt.show()

plot_imbalance_deep_dive(results_all, label="H2=4.5", n_days=3, start_day=4)




    # ── Storage analysis plot ─────────────────────────────────────────────────
# dt = 0.25

# storage_full  = np.concatenate([d["storage"]   for d in results_all])
# h_prod_full = np.concatenate([d["h_RT"] for d in results_all]) * dt
# h_sell_full = np.concatenate([d["h_sell_RT"] for d in results_all]) * dt
# h_sell_full   = np.concatenate([d["h_sell_RT"] for d in results_all]) * dt
# wind_full     = np.concatenate([d["wind_real"] for d in results_all])
# p_DA_full     = np.concatenate([d["price_DA"]  for d in results_all])

# T_total = len(storage_full)
# t_days  = np.arange(T_total) * dt / 24

# # daily aggregates
# daily_prod = h_prod_full.reshape(num_days, 96).sum(axis=1)
# daily_sell = h_sell_full.reshape(num_days, 96).sum(axis=1)
# days_idx   = np.arange(num_days)

# fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
# fig.suptitle("Hydrogen storage", fontsize=13)

# # Panel 1: storage level over time
# ax = axes[0]
# ax.fill_between(t_days, 0, storage_full / 1000,
#                 alpha=0.4, color="steelblue")
# ax.plot(t_days, storage_full / 1000,
#         color="steelblue", lw=1.0, label="Storage level")
# ax.axhline(storage_max / 1000, color="red",    lw=1.2, ls="--",
#            label=f"Capacity ({storage_max/1000:.1f} t)")
# ax.axhline(h_min_daily / 1000, color="orange", lw=1.0, ls=":",
#            label=f"Daily minimum ({h_min_daily/1000:.1f} t)")
# ax.set_ylabel("Tonnes H₂")
# ax.set_xlabel("Day of simulation")
# ax.set_title("Storage level")
# ax.legend(fontsize=8); ax.grid(alpha=0.3)
# ax.set_ylim(bottom=0)

# # Panel 2: daily production vs sales
# ax = axes[1]
# ax.bar(days_idx, daily_prod / 1000,
#        color="steelblue", alpha=0.7, label="Produced (t/day)")
# ax.bar(days_idx, -daily_sell / 1000,
#        color="orangered",  alpha=0.7, label="Sold (t/day)")
# ax.axhline(h_min_daily / 1000,  color="orange", lw=1.2, ls="--",
#            label=f"Min ({h_min_daily/1000:.1f} t/day)")
# ax.axhline(-h_min_daily / 1000, color="orange", lw=1.2, ls="--")
# ax.axhline(0, color="black", lw=0.5)
# ax.set_ylabel("Tonnes H₂ / day")
# ax.set_xlabel("Day")
# ax.set_title("Daily H₂ production vs sales")
# ax.legend(fontsize=8); ax.grid(alpha=0.3)

# # Panel 3: days contract met
# ax = axes[2]
# contract_met = daily_sell >= h_min_daily
# colors_c = ["steelblue" if m else "red" for m in contract_met]
# ax.bar(days_idx, daily_sell / 1000, color=colors_c, alpha=0.8)
# ax.axhline(h_min_daily / 1000, color="black", lw=1.5, ls="--",
#            label=f"Contract min ({h_min_daily/1000:.1f} t/day)")
# ax.set_ylabel("H₂ sold (t/day)")
# ax.set_xlabel("Day")
# ax.set_title("Daily H₂ sales vs contract minimum  (red = below minimum)")
# ax.legend(fontsize=8); ax.grid(alpha=0.3)

# plt.tight_layout()
# plt.savefig("figures/storage_analysis.png", dpi=150, bbox_inches="tight")
# plt.show()

# # summary stats
# n_contract = int(np.sum(contract_met))
# mean_fill  = float(np.mean(storage_full / storage_max * 100))
# n_full     = int(np.sum(storage_full >= storage_max * 0.99))
# n_empty    = int(np.sum(storage_full <= storage_max * 0.01))

# print(f"\n=== STORAGE SUMMARY ===")
# print(f"  Storage capacity       : {storage_max/1000:.1f} t")
# print(f"  Daily minimum contract : {h_min_daily/1000:.1f} t/day")
# print(f"  Mean fill level        : {mean_fill:.1f}%")
# print(f"  Timesteps at capacity  : {n_full} ({100*n_full/T_total:.1f}%)")
# print(f"  Timesteps empty        : {n_empty} ({100*n_empty/T_total:.1f}%)")
# print(f"  Days contract met      : {n_contract}/{num_days} "
#       f"({100*n_contract/num_days:.1f}%)")
# print(f"  Total H2 produced      : {h_prod_full.sum()/1000:.1f} t")
# print(f"  Total H2 sold          : {h_sell_full.sum()/1000:.1f} t")
# print(f"  Net into storage       : "
#       f"{(h_prod_full.sum()-h_sell_full.sum())/1000:+.1f} t")

# fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=False)
# fig.suptitle("Hydrogen storage", fontsize=13)

# # Panel 1: storage level
# ax = axes[0]
# ax.fill_between(t_days, 0, storage_full / 1000,
#                 alpha=0.4, color="steelblue")
# ax.plot(t_days, storage_full / 1000,
#         color="steelblue", lw=1.0, label="Storage level")
# ax.axhline(storage_max / 1000, color="red",    lw=1.2, ls="--",
#            label=f"Capacity ({storage_max/1000:.1f} t)")
# ax.axhline(h_min_daily / 1000, color="orange", lw=1.0, ls=":",
#            label=f"Daily minimum ({h_min_daily/1000:.1f} t)")
# ax.set_ylabel("Tonnes H₂")
# ax.set_xlabel("Day of simulation")
# ax.set_title("Storage level")
# ax.legend(fontsize=8); ax.grid(alpha=0.3)
# ax.set_ylim(bottom=0)

# # Panel 2: cumulative production vs sales
# ax = axes[1]
# cum_prod = np.cumsum(h_prod_full) / 1000   # tonnes
# cum_sell = np.cumsum(h_sell_full) / 1000   # tonnes
# ax.plot(t_days, cum_prod, color="steelblue", lw=1.5, label="Cumulative produced")
# ax.plot(t_days, cum_sell, color="orangered",  lw=1.5, label="Cumulative sold")
# ax.fill_between(t_days, cum_sell, cum_prod,
#                 alpha=0.2, color="steelblue",
#                 label="In storage (produced − sold)")
# ax.set_ylabel("Tonnes H₂ (cumulative)")
# ax.set_xlabel("Day of simulation")
# ax.set_title("Cumulative H₂ production vs sales")
# ax.legend(fontsize=8); ax.grid(alpha=0.3)

# # Panel 3: daily production vs sales bar
# ax = axes[2]
# ax.bar(days_idx, daily_prod / 1000,
#        color="steelblue", alpha=0.7, label="Produced (t/day)")
# ax.bar(days_idx, -daily_sell / 1000,
#        color="orangered",  alpha=0.7, label="Sold (t/day)")
# ax.axhline(h_min_daily / 1000,  color="orange", lw=1.2, ls="--",
#            label=f"Min ({h_min_daily/1000:.1f} t/day)")
# ax.axhline(-h_min_daily / 1000, color="orange", lw=1.2, ls="--")
# ax.axhline(0, color="black", lw=0.5)
# ax.set_ylabel("Tonnes H₂ / day")
# ax.set_xlabel("Day")
# ax.set_title("Daily H₂ production vs sales")
# ax.legend(fontsize=8); ax.grid(alpha=0.3)

# # Panel 4: contract compliance
# ax = axes[3]
# contract_met = daily_sell >= h_min_daily
# colors_c = ["steelblue" if m else "red" for m in contract_met]
# ax.bar(days_idx, daily_sell / 1000, color=colors_c, alpha=0.8)
# ax.axhline(h_min_daily / 1000, color="black", lw=1.5, ls="--",
#            label=f"Contract ({h_min_daily/1000:.1f} t/day)")
# ax.set_ylabel("H₂ sold (t/day)")
# ax.set_xlabel("Day")
# ax.set_title("Daily H₂ sales vs contract  (red = below minimum)")
# ax.legend(fontsize=8); ax.grid(alpha=0.3)

# plt.tight_layout()
# plt.savefig("figures/storage_analysis.png", dpi=150, bbox_inches="tight")
# plt.show()