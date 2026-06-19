from kiwisolver import Constraint

from Piecewise import generate_piecewise
import Hydrogen_production as hp
import numpy as np
import pandas as pd
from Data_preparation import DataPreparationCSV
from Optimizer import InputData, ElectrolyzerModel
from Economics import EconomicsAnalysis, EconomicsConfig

#from Sensitivity import SizingStudy, SizingConfig
import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
#from matplotlib.colors import TwoSlopeNorm




USE_STOCHASTIC = False
N_SCENARIOS    = 20
N_NEIGHBORS    = 50

from Scenario_gen import compute_forecast_errors, generate_scenarios
from StochasticOptimizer import StochasticInputData, StochasticElectrolyzerModel



# --- Use ERA5 + power curve instead ---
prep = DataPreparationCSV(
    start_date="2025-03-05",
    end_date="2026-03-05",
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


# Data
wind_real = wind_df['wind_kW'].values  # Wind power [kW]
price_real = el_price_df['el_price[EUR/MWh]'].values  # Day-ahead price [EUR/MWh]
price_H2 = np.full(len(wind_real), 4.5) #EUR/kg
#price_H2 = np.full(H, 5.5)  # Hydrogen price [EUR/kg], constant for now, from European Hydrogen Observatory
price_FCR = fcr_df['price_FCR'].values  # FCR price [EUR/kW]
FCR_volume = fcr_df['FCR_volume'].values # FCR volume [kW], using "up" as example
grid_cap = 80000.0  # Max grid power [kW], adjust as needed
h_min_daily = 0.0  # Minimum daily hydrogen production [kg], adjust as needed
h_sell_max =200000.0   # Maximum sell rate per timestep [kg], adjust as needed

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
P_min = 0.2*P_max     # Minimum power [kW] #CHANGE?
P_sb = min(P_stack)     # Standby power [kW] #CHANGE###

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
storage_max =0.0  #50000   # Max storage [kg]
storage_initial = 0.0     # Initial storage [kg]
N_down = 6             # Downtime in 15min intervals, adjust as needed

t0 = 0

storage_state = storage_initial
z_on_state = 0
P_prev = 0

TIMESTEPS_PER_DAY = 96
num_days =  364#len(wind_real) // TIMESTEPS_PER_DAY

blocks_per_day = TIMESTEPS_PER_DAY // 16


RUN_MODE = "seasonal"   # ← change to "full" to run the whole year
 
# Start date of the simulation (must match your DataPreparation start_date)
SIM_START = pd.Timestamp("2025-03-05")
 
# ── Build the list of day indices to simulate ─────────────────────────────────
 
def _seasonal_day_indices(sim_start: pd.Timestamp, n_total_days: int,
                           weeks_per_season: int = 2) -> list[int]:
    """
    Return a sorted list of day indices (0-based from sim_start) covering
    `weeks_per_season` weeks centred on the middle of each season:
      Winter  → mid-January  (15 Jan)
      Spring  → mid-April    (15 Apr)
      Summer  → mid-July     (15 Jul)
      Autumn  → mid-October  (15 Oct)
 
    If the target window falls outside the simulation range it is skipped,
    so a 1-year run will always capture all four seasons.
    """
    half = (weeks_per_season * 7) // 2   # days on each side of the centre
 
    # Season centres — try both the year of sim_start and the following year
    # so that a run starting in March still captures January of the next year.
    centres = []
    for year in [sim_start.year, sim_start.year + 1]:
        for month, day in [(1, 15), (4, 15), (7, 15), (10, 15)]:
            centres.append(pd.Timestamp(year, month, day))
 
    indices = set()
    sim_end = sim_start + pd.Timedelta(days=n_total_days - 1)
 
    for centre in centres:
        window_start = centre - pd.Timedelta(days=half)
        window_end   = centre + pd.Timedelta(days=half - 1)
 
        # Clip to simulation range
        clipped_start = max(window_start, sim_start)
        clipped_end   = min(window_end,   sim_end)
 
        if clipped_start > clipped_end:
            continue   # window entirely outside simulation range
 
        for dt in pd.date_range(clipped_start, clipped_end, freq="D"):
            idx = (dt - sim_start).days
            if 0 <= idx < n_total_days:
                indices.add(idx)
 
    return sorted(indices)
 
 
# Total number of days in the dataset
N_TOTAL_DAYS = len(wind_real) // TIMESTEPS_PER_DAY
 
if RUN_MODE == "full":
    days_to_run = list(range(N_TOTAL_DAYS))
    print(f"Run mode: FULL YEAR ({N_TOTAL_DAYS} days)")
 
elif RUN_MODE == "seasonal":
    days_to_run = _seasonal_day_indices(SIM_START, N_TOTAL_DAYS, weeks_per_season=2)
    print(f"Run mode: SEASONAL (2 weeks × 4 seasons = {len(days_to_run)} days)")
    # Show which calendar dates are included
    dates = [SIM_START + pd.Timedelta(days=i) for i in days_to_run]
    seasons = {"Jan": [], "Apr": [], "Jul": [], "Oct": []}
    for d in dates:
        if   d.month == 1:  seasons["Jan"].append(d.strftime("%d %b %Y"))
        elif d.month == 4:  seasons["Apr"].append(d.strftime("%d %b %Y"))
        elif d.month == 7:  seasons["Jul"].append(d.strftime("%d %b %Y"))
        elif d.month == 10: seasons["Oct"].append(d.strftime("%d %b %Y"))
    for season, ds in seasons.items():
        if ds:
            print(f"  {season}: {ds[0]}  →  {ds[-1]}  ({len(ds)} days)")
else:
    raise ValueError(f"Unknown RUN_MODE '{RUN_MODE}'. Use 'full' or 'seasonal'.")
 
num_days = len(days_to_run)
 
season_starts = {days_to_run[0]}
for i in range(1, len(days_to_run)):
    if days_to_run[i] - days_to_run[i-1] > 1:   # gap in days
        season_starts.add(days_to_run[i])


results_all = []
DA_objectives = []
RT_objectives = []

for loop_idx, day in enumerate(days_to_run):
 
    print(f"Processing day {loop_idx + 1} of {num_days}  "
          f"(day index {day}  =  "
          f"{(SIM_START + pd.Timedelta(days=day)).strftime('%d %b %Y')})...")
    

 
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

    if day in season_starts:
        storage_state  = storage_initial
        z_on_state     = 0
        P_prev         = 0
        print(f"  → New season window, state reset.")

    # --- 4. Build model ---
    if USE_STOCHASTIC:
        price_sc, wind_sc, imb_sc, probs = generate_scenarios(
        price_fc_day, wind_fc_day, imbalance_fc_day,
        price_errors, wind_errors, imb_errors, wind_forecasts,
        n_scenarios=N_SCENARIOS, n_neighbors=N_NEIGHBORS,
        mode="random",        
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
    P_buy_DA = np.array(DA_model.results.P_buy)
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
    P_buy_RT  = np.array(RT_model.results.P_buy)

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
    results_all.append({
        "P_DA": P_DA,
        "P_grid": P_grid,
        "h_sell_DA": h_sell_DA,
        "h_sell_RT": h_sell_RT,
        "P_curt_DA": P_curt_DA,
        "P_buy_RT": P_buy_RT,
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
        "price_FCR":   price_FCR_day,        #EUR/kW
        "date": (SIM_START + pd.Timedelta(days=day)).strftime("%Y-%m-%d")
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
 
    discount_rate               = 0.08,
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
P_buy_RT_full  = np.concatenate([d["P_buy_RT"]  for d in results_all])
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
cost_buy     = np.sum(P_buy_RT_full  * price_real_full * dt / 1000)
rev_FCR      = np.sum(DA_FCR_full * price_FCR_full * dt_FCR)   # kW * EUR/kW/h * h = EUR

total = rev_elec_DA + rev_imb + rev_h2_RT - cost_buy + rev_FCR

print("\n=== REALIZED REVENUE BREAKDOWN ===")
print(f"  DA electricity  : {rev_elec_DA:>12,.0f} EUR")
print(f"  Imbalance       : {rev_imb:>12,.0f} EUR")
print(f"  RT hydrogen     : {rev_h2_RT:>12,.0f} EUR")
print(f"  Grid purchase   : {-cost_buy:>12,.0f} EUR")
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
lhs = wind_real_full + P_buy_RT_full
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

from plots_annual import plot_annual_results, plot_h2_sensitivity, override_h2_price

plot_annual_results(results_all, econ_results, P_max, wind_capacity_kw,
                    price_H2=price_H2[0], save_dir="figures/")

# H2 sensitivity — no new solver calls
plot_h2_sensitivity(results_all, cfg, h2_prices=[3.0, 4.5, 6.0, 8.0],
                    P_el_kw=P_max, save_dir="figures/")