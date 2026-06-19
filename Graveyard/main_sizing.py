from kiwisolver import Constraint

from Piecewise import generate_piecewise
import Hydrogen_production as hp
import numpy as np
from Data_preparation import DataPreparationCSV
from Optimizer import InputData, ElectrolyzerModel
from Economics import EconomicsAnalysis, EconomicsConfig
from Sensitivity import SizingStudy, SizingConfig


#from Sensitivity import SizingStudy, SizingConfig
import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
#from matplotlib.colors import TwoSlopeNorm




USE_STOCHASTIC = False
N_SCENARIOS    = 20
N_NEIGHBORS    = 50

from Scenario_gen import compute_forecast_errors, generate_scenarios
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
    N_stacks=40*20,
    N_cell=250,
    n_segments=3,
    plot=True  # Set to True to see plot
)

print("Piecewise approximation generated.")
print("A_s:", A_s)
print("Typical efficiency (kg/kWh):", A_s)
P_stack_base = P_stack.copy() 
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
grid_cap = 40000.0  # Max grid power [kW], adjust as needed
h_min_daily = 0.0  # Minimum daily hydrogen production [kg], adjust as needed
h_sell_max =200000.0   # Maximum sell rate per timestep [kg], adjust as needed

# Electrolyzer parameters
P_total, m_dot, P_stack = hp.electrolyzer_h2_prod_curve(
    T=80,
    I_range=I_range,
    A_cell=0.0625,
    p=40,
    p_out=45,
    N_stacks=40*20,
    N_cell=250
)

P_max = max(P_stack) # Maximum power [kW] (adjust based on system)
P_min = 0.2*P_max      # Minimum power [kW] #CHANGE?
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
N_down = 3             # Downtime in 15min intervals, adjust as needed

t0 = 0

storage_state = storage_initial
z_on_state = 0
P_prev = 0

TIMESTEPS_PER_DAY = 96
num_days =  15#len(wind_real) // TIMESTEPS_PER_DAY

blocks_per_day = TIMESTEPS_PER_DAY // 16


results_all = []
DA_objectives = []
RT_objectives = []


def run_one_size(P_el_kw: float):
    """
    Run the full rolling-horizon DA+RT optimisation for a given electrolyzer
    capacity and return the per-day results list.
 
    This is essentially your existing day-loop, with P_max / P_min / P_sb /
    piecewise coefficients recomputed for the new capacity.
    """
 
    # ── 1. Recompute piecewise model at new capacity ──────────────────────────
    # Scale the number of stacks to hit the target capacity.
    # N_stacks_base is the reference you used in the original main.py.
    N_stacks_base = 40 * 20           # same as original
    P_max_base    = max(P_stack_base)  # from the original piecewise run (global)
    scale         = P_el_kw / P_max_base
    N_stacks_new  = max(1, round(N_stacks_base * scale))
 
    A_s_new, B_s_new, P_seg_min_new, P_seg_max_new, \
    P_total_new, m_dot_new, P_stack_new = generate_piecewise(
        hp,
        T=80,
        I_range=I_range,              # global, unchanged
        A_cell=0.0625,
        p=40,
        p_out=300,
        N_stacks=N_stacks_new,
        N_cell=250,
        n_segments=3,
        plot=False,
    )
 
    P_min_new = min(P_stack_new)
    P_max_new = max(P_stack_new)
    P_sb_new  = min(P_stack_new)
 
    # ── 2. Run the day loop ───────────────────────────────────────────────────
    # Copy-paste your existing day-loop here, substituting all references to
    # P_min, P_max, P_sb, A_s, B_s, P_seg_min, P_seg_max with the *_new variants.
    # Everything else (data slices, prices, storage, z_on_state, etc.) is unchanged.
 
    local_results_all  = []
    local_storage      = storage_initial   # reset storage for each sizing run
    local_z_on_state   = 0
    local_P_prev       = 0
 
    for day in range(num_days):
        start = day * TIMESTEPS_PER_DAY
        end   = (day + 1) * TIMESTEPS_PER_DAY
        start_block = day * blocks_per_day
        end_block   = (day + 1) * blocks_per_day
 
        price_FCR_day   = price_FCR[start_block:end_block]
        FCR_volume_day  = FCR_volume[start_block:end_block]
        wind_fc_day     = wind_forecast[start:end]
        price_fc_day    = price_forecast[start:end]
        price_H2_day    = price_H2[start:end]
        imbalance_fc_day = imbalance_forecast[start:end]
        wind_rt_day     = wind_real[start:end]
        price_rt_day    = price_real[start:end]
        imbalance_day   = imbalance_price[start:end]
        price_H2_RT     = price_H2[start:end]

        # DA model
        input_DA = InputData(
            T=TIMESTEPS_PER_DAY, S=len(A_s_new), I=len(price_FCR_day),
            wind=None, wind_forecast=wind_fc_day,
            price_DA=price_fc_day, price_H2=price_H2_day,
            price_FCR=price_FCR_day, r_FCR_DA=None, imbalance_price=None,
            P_min=P_min_new, P_max=P_max_new, P_sb=P_sb_new,
            P_seg_min=P_seg_min_new, P_seg_max=P_seg_max_new,
            A_s=A_s_new, B_s=B_s_new,
            storage_max=storage_max, storage_initial=local_storage,
            h_min_daily=h_min_daily, h_sell_max=h_sell_max,
            grid_cap=grid_cap, N_down=N_down,
            FCR_volume=FCR_volume_day,
            z_on_prev=local_z_on_state, P_prev=local_P_prev,
            is_DA=True, DA_export=None,
        )
        DA_model = ElectrolyzerModel(input_DA)
        DA_model.run()
 
        P_DA      = np.array(DA_model.results.P_DA)
        r_FCR_DA  = DA_model.results.r_FCR_block
 
        # RT model
        input_RT = InputData(
            T=TIMESTEPS_PER_DAY, S=len(A_s_new), I=len(price_FCR_day),
            wind=wind_rt_day, wind_forecast=wind_fc_day,
            price_DA=price_rt_day, price_H2=price_H2_RT,
            price_FCR=price_FCR_day, r_FCR_DA=r_FCR_DA,
            imbalance_price=imbalance_day,
            P_min=P_min_new, P_max=P_max_new, P_sb=P_sb_new,
            P_seg_min=P_seg_min_new, P_seg_max=P_seg_max_new,
            A_s=A_s_new, B_s=B_s_new,
            storage_max=storage_max, storage_initial=local_storage,
            h_min_daily=h_min_daily, h_sell_max=h_sell_max,
            grid_cap=grid_cap, N_down=N_down,
            FCR_volume=FCR_volume_day,
            z_on_prev=local_z_on_state, P_prev=local_P_prev,
            is_DA=False, DA_export=P_DA,
        )
        RT_model = ElectrolyzerModel(input_RT)
        RT_model.run()
 
        # Update rolling state
        local_storage    = RT_model.results.storage[-1]
        local_z_on_state = RT_model.results.z_on[-1]
        local_P_prev     = RT_model.results.P_elec[-1]
 
        local_results_all.append({
            "P_DA":        P_DA,
            "P_grid":      np.array(RT_model.results.P_grid),
            "h_sell_RT":   np.array(RT_model.results.h_sell),
            "DA_FCR":      r_FCR_DA,
            "imbalance":   np.array(RT_model.results.P_grid) - P_DA,
            "wind_real":   wind_rt_day,
            "P_curt_RT":   np.array(RT_model.results.P_curt),
            "P_elec_RT":   np.array(RT_model.results.P_elec),
            "P_buy_RT":    np.array(RT_model.results.P_buy),
            "price_DA":    price_rt_day,
            "price_imb":   imbalance_day,
            "price_H2":    price_H2_RT,
            "price_FCR":   price_FCR_day,
        })
 
    return local_results_all
 


base_econ_cfg = EconomicsConfig(
    electrolyzer_capacity_kw    = P_max,           # overridden per sweep step
    wind_capacity_kw            = wind_capacity_kw,
    storage_capacity_kg         = storage_max,
 
    capex_electrolyzer_eur_kw   = 1_000.0,
    capex_wind_eur_kw           = 2_000.0,
    capex_storage_eur_kg        = 500.0,
 
    opex_electrolyzer_eur_kw_yr = 20.0,
    opex_wind_eur_kw_yr         = 50.0,
 
    discount_rate               = 0.08,
    project_lifetime_yr         = 20,
    simulation_years            = num_days / 365.0,
    tax_rate                    = 0.22,
)
 
sizing_cfg = SizingConfig(
    ratios           = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00],
    wind_capacity_kw = wind_capacity_kw,
    econ_config_base = base_econ_cfg,
    # Uncomment to apply economy-of-scale CAPEX curve:
    # capex_scale_fn = lambda p: 1_000.0 * (p / 100_000.0) ** (-0.15),
)
 
study     = SizingStudy(sizing_cfg)
sizing_df = study.run(run_one_size)
study.print_summary(sizing_df)
study.plot(sizing_df, save_path="sizing_results.png")
 
# H2 price sensitivity — runs economics post-hoc, no extra solver calls
study.run_h2_sensitivity(
    run_one_size_fn = run_one_size,
    h2_prices       = [3.0, 4.5, 6.0, 8.0],
    save_path       = "sizing_sensitivity.png",
)
 