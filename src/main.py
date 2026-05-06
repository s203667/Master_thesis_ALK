from kiwisolver import Constraint

from Piecewise import generate_piecewise
import Hydrogen_production as hp
import numpy as np
from Data_preparation import DataPreparationCSV
from Optimizer import InputData, ElectrolyzerModel
import matplotlib.pyplot as plt

# --- DATA PREPARATION --- #
print("Preparing data...")
prep = DataPreparationCSV(
    start_date="2025-03-05",
    end_date="2025-03-09",
    el_price_file_name="DayAheadPrices.csv", 
    wind_file_name="WindProd5MinRealtime.csv",  # Update if data_prep changes
    fcr_file_name="FcrDK1.csv",  
    imbalance_file_name="ImbalancePrice.csv"
)
data = prep.data

wind_df = data['wind']
el_price_df = data['el_price']
fcr_df = data['fcr']
imbalance_df = data['imbalance']
imbalance_price = imbalance_df['ImbalancePriceEUR'].values

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
    p=40,
    p_out=45,
    N_stacks=40*20,
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
H = 96  # e.g. 1 day (96 × 15min)
step = 24  # move horizon by 6 hours
S = len(A_s)      # Number of segments
I = len(fcr_df)   # Number of FCR blocks (4-hour intervals)

# Data
wind = wind_df['wind_kW'].values  # Wind power [kW]
price_DA = el_price_df['el_price[EUR/MWh]'].values  # Day-ahead price [EUR/MWh]
price_H2 = np.full(H, 5.5)  # Hydrogen price [EUR/kg], constant for now, from European Hydrogen Observatory
price_FCR = fcr_df['price_FCR'].values  # FCR price [EUR/kW]
FCR_volume = fcr_df['FCR_volume'].values # FCR volume [kW], using "up" as example
grid_cap = 4100.0  # Max grid power [kW], adjust as needed
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
P_min = min(P_stack)      # Minimum power [kW] #CHANGE###
P_max = max(P_stack) # Maximum power [kW] (adjust based on system)
P_sb = 650.0     # Standby power [kW] #CHANGE###

#Real data
wind_real = wind
price_real = price_DA

#Add "Forecast" that is just the day before repeated
lag = 96  # 1 day

wind_forecast = np.roll(wind_real, lag)
price_forecast = np.roll(price_real, lag)

# For the first day, fill with actual values (or zeros)
wind_forecast[:lag] = wind_real[:lag]
price_forecast[:lag] = price_real[:lag]

print("NaNs in FCR price:", np.isnan(price_FCR).sum())
print("NaNs in FCR volume:", np.isnan(FCR_volume).sum())

#r_FCR_block = 0  ####TEMPORARILY WHILE TESTING #####
# Storage parameters
storage_max = 2000.0     # Max storage [kg]
storage_initial = 0.0     # Initial storage [kg]
N_down = 3             # Downtime in 15min intervals, adjust as needed

t0 = 0

storage_state = storage_initial
z_on_state = 0
P_prev = 0

results_all = []

# --- SINGLE-WINDOW DEBUG MODEL (192 steps) ---

# h_full = None
# # Full horizon = all timesteps in your data (here: 2016-01-01 to 2016-01-03)
# H_eff = len(wind)          # should be 192 for 2 days at 15-min resolution
# S = len(A_s)

# # Time-dependent slices
# wind_slice_real = wind_real[t0:t0 + step_eff]
# price_slice_real = price_real[t0:t0 + step_eff]
# wind_slice_forecast = wind_forecast[t0:t0 + H_eff]
# price_slice_forecast = price_forecast[t0:t0 + H_eff]
# h2_price_slice = np.full(H_eff, 5.5)

# # FCR blocks (4-hour blocks = 16 timesteps)
# i0 = 0
# I_eff = int(np.ceil(H_eff / 16))
# price_FCR_slice = price_FCR[i0:i0 + I_eff]
# FCR_volume_slice = FCR_volume[i0:i0 + I_eff]
# I_eff = len(price_FCR_slice)  # just to be safe

# # Initial states
# storage_state = storage_initial
# z_on_state = 1
# P_prev = 0

# input_data = InputData(
#     T=H_eff,
#     S=S,
#     I=I_eff,
#     wind = wind_slice_forecast,
#     price_DA = price_slice_forecast,
#     price_H2=h2_price_slice,
#     price_FCR=price_FCR_slice,
#     P_min=P_min,
#     P_max=P_max,    
#     P_sb=P_sb,
#     P_seg_min=P_seg_min,
#     P_seg_max=P_seg_max,
#     A_s=A_s,
#     B_s=B_s,
#     storage_max=storage_max,
#     storage_initial=storage_state,
#     grid_cap=grid_cap,
#     N_down=N_down,
#     FCR_volume=FCR_volume_slice,
#     z_on_prev=z_on_state,
#     P_prev=P_prev,
# )

# model_full = ElectrolyzerModel(input_data)
# model_full.run()

# step_eff = min(step, H_eff)
# storage_state = model_full.results.storage[step_eff - 1]

# if getattr(model_full.results, "has_solution", False):
#     print("Single-window model feasible")
#     print("Total H2:", sum(model_full.results.h))

#     print("First 20 z_on:", model_full.results.z_on[:20])
#     print("First 20 storage:", model_full.results.storage[:20])
#     print("First 20 H2:", model_full.results.h[:20])
# else:
#     step_eff = min(step, H_eff)
#     storage_state = model_full.results.storage[step_eff - 1]
#     print("Single-window model feasible")
#     print("Total H2:", sum(model_full.results.h))

# if model_full.results.has_solution:
#     h_full = np.array(model_full.results.h)

# h_roll = None

while t0 < len(wind):

    # --- 1. Effective horizon ---
    H_eff = min(H, len(wind) - t0)

    # --- 1b. Implemented part ---
    step_eff = min(step, H_eff)

    # --- 2. Slice ALL time-dependent data consistently ---
    wind_slice_real = wind_real[t0:t0 + H_eff]
    price_slice_real = price_real[t0:t0 + H_eff]
    wind_slice_forecast = wind_forecast[t0:t0 + H_eff]
    price_slice_forecast = price_forecast[t0:t0 + H_eff]
    h2_price_slice = np.full(H_eff, 5.5)
    imbalance_slice = imbalance_price[t0:t0 + H_eff]

    # --- 3. Handle block data (FCR) carefully ---
    i0 = t0 // 16
    I_eff = int(np.ceil(H_eff / 16))

    price_FCR_slice = price_FCR[i0:i0 + I_eff]
    FCR_volume_slice = FCR_volume[i0:i0 + I_eff]

    # Ensure consistency
    I_eff = len(price_FCR_slice)

    # --- 4. Build model ---
    input_data = InputData(
        T=H_eff,
        S=S,
        I=I_eff,
        wind=wind_slice_real,
        wind_forecast=wind_slice_forecast,
        price_DA=price_slice_forecast,
        price_H2=h2_price_slice,
        price_FCR=price_FCR_slice,
        P_min=P_min,
        P_max=P_max,
        P_sb=P_sb,
        P_seg_min=P_seg_min,
        P_seg_max=P_seg_max,
        A_s=A_s,
        B_s=B_s,
        storage_max=storage_max,
        storage_initial=storage_state,
        grid_cap=grid_cap,
        N_down=N_down,
        FCR_volume=FCR_volume_slice,
        imbalance_price=imbalance_slice,
        z_on_prev=z_on_state,
        P_prev=P_prev
    )

    model = ElectrolyzerModel(input_data)
    model.run()
    print(f"\n--- Window starting at t0={t0} ---")
    print(f"  storage_initial passed to model: {storage_state}")
    
    if not model.results.has_solution:
        print("Single-window model infeasible at t0=0")
        model.debug_constraints()  # see below
        break
        # --- 5. Extract only implemented part (step) ---
    step_eff = min(step, H_eff)

    storage_state = model.results.storage[step_eff - 1]
    z_on_state = model.results.z_on[step_eff - 1]
    P_prev = sum(
        model.model.P_seg[step_eff - 1, s].value
        for s in model.model.S
    )

    if model.results.has_solution:
        step_eff = min(step, H_eff)
        storage_state = model.results.storage[step_eff - 1]
        print(f"  storage at end of window: {storage_state}")


    results_all.append({
        "h": model.results.h[:step_eff],
        "storage": model.results.storage[:step_eff],
        "z_on": model.results.z_on[:step_eff],
        "P_buy": model.results.P_buy[:step_eff],
        "P_sell": model.results.P_sell[:step_eff],
        "P_curt": model.results.P_curt[:step_eff],
        "r_FCR_block": model.results.r_FCR_block[:step_eff],
        "wind_real": wind_slice_real,
        "price_real": price_slice_real,
        "wind_forecast": wind_slice_forecast[:step_eff],
        "price_forecast": price_slice_forecast[:step_eff],
    })
    # else:
    #     print(f"Warning: Model infeasible at window starting at t0={t0}, skipping this window.")
    #     break
    #     # Optionally, handle infeasible cases (e.g., set default values or break)

    # --- 6. Move forward ---
    t0 += step_eff


    

# Access results
print("Hydrogen production:", [h for result in results_all for h in result["h"][:5]])
  # First 5 values
print("Storage levels:", [storage for result in results_all for storage in result["storage"][:5]])
print("wind power last ten periods:", wind[:10])
print("Maximum wind power:", np.max(wind))
print("Final t0:", t0)
print("Expected t0:", len(wind))

print(f"price_FCR_slice length: {len(price_FCR_slice)}")
print(f"FCR_volume_slice length: {len(FCR_volume_slice)}")
# Debug: Check actual z_on values
print("\n--- DEBUG ---")
print(f"Number of results: {len(results_all)}")
if results_all:
    print(f"First result z_on values (first 20):", results_all[0]["z_on"][:20])
    print(f"First result z_on unique values:", set(results_all[0]["z_on"]))
    print(f"First result z_on dtype: {type(results_all[0]['z_on'][0])}")

def reconstruct_full_timeseries(results_all, step):
    all_h = []
    all_storage = []
    all_z_on = []
    all_p_buy = []
    all_p_sell = []
    all_fcr_blocks = []

    for i, result in enumerate(results_all):
        # Determine how many steps this window contributes
        if i == len(results_all) - 1:
            H_res = len(result["h"])   # last window: take all
        else:
            H_res = step            # earlier windows: take only first `step`

        # Append exactly the used part
        all_h.extend(result["h"][:H_res])
        all_storage.extend(result["storage"][:H_res])
        all_z_on.extend(result["z_on"][:H_res])
        all_p_buy.extend(result["P_buy"][:H_res])
        all_p_sell.extend(result["P_sell"][:H_res])

        # Expand FCR blocks to 16 timesteps each
        for block_value in result["r_FCR_block"]:
            all_fcr_blocks.extend([block_value] * 16)

    return (
        np.array(all_h),
        np.array(all_storage),
        np.array(all_z_on),
        np.array(all_p_buy),
        np.array(all_p_sell),
        np.array(all_fcr_blocks),
    )

def downsample(arr, factor=20):
    return arr[::factor]


# --- RECONSTRUCT FULL YEAR ---
all_h, all_storage, all_z_on, all_p_buy, all_p_sell, all_fcr = \
    reconstruct_full_timeseries(results_all, step)

h_roll = all_h.copy()
print(f"Full timeline length: {len(all_h)} timesteps")

# --- DEBUG CHECK ---
print("First 20 z_on:", all_z_on[:20])
print("Unique z_on values:", set(all_z_on[:500]))

# --- SUMMARY ---
print("\nFirst 500 timesteps summary:")
print(f"Total H2 production: {sum(all_h[:500]):.1f} kg")
print(f"On periods: {sum(all_z_on[:500])} out of 500")

# --- PLOTS (first 500 timesteps) ---
plt.figure(figsize=(12,8))

plt.subplot(3,1,1)
plt.plot(all_z_on, linewidth=0.8)
plt.title("Electrolyzer On/Off (first 500 timesteps)")
plt.grid(True, alpha=0.3)

plt.subplot(3,1,2)
plt.plot(all_h, linewidth=0.8)
plt.title("H2 Production (kg/h)")
plt.grid(True, alpha=0.3)

plt.subplot(3,1,3)
plt.plot(all_storage, linewidth=0.8)
plt.title("Storage Level (kg)")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- FULL-YEAR PLOTS (DOWNSAMPLED) ---
plt.figure(figsize=(14,10))

# plt.subplot(4,1,1)
# plt.plot(downsample(all_z_on, 20))
# plt.title("Electrolyzer On/Off (downsampled)")
# plt.grid(True, alpha=0.3)

plt.subplot(4,1,1)
plt.plot(downsample(all_h, 20))
plt.plot(downsample(all_storage, 20))
plt.legend(["H2 Production", "Storage Level"])
plt.title("H2 Production (kg/h, downsampled)")
plt.grid(True, alpha=0.3)

plt.subplot(4,1,2)
plt.plot(downsample(all_p_buy, 20), label="Buy")
plt.plot(downsample(all_p_sell, 20), label="Sell")
plt.legend()
plt.title("Electricity Trading (downsampled)")
plt.grid(True, alpha=0.3)

plt.subplot(4,1,3)
plt.plot(downsample(all_fcr, 20))
plt.title("FCR Power (downsampled)")
plt.grid(True, alpha=0.3)


plt.tight_layout()
plt.show()

wind_real_full = np.concatenate([r["wind_real"] for r in results_all])
wind_forecast_full = np.concatenate([r["wind_forecast"] for r in results_all])

forecast_error = wind_real_full - wind_forecast_full

plt.plot(forecast_error)
plt.title("Wind forecast error")
plt.xlabel("Timestep")
plt.ylabel("Forecast error (kW)")
plt.grid(True, alpha=0.3)
plt.show()

# --- COMPUTE FULL-SIMULATION OBJECTIVE ---
dt = 0.25          # 15 minutes
dt_FCR = 4         # FCR blocks are 4 hours

# Slice price arrays to match lengths
price_H2_full = price_H2[:len(all_h)]
price_DA_full = price_DA[:len(all_h)]

# FCR price must match number of blocks
price_FCR_full = price_FCR[:len(all_fcr)]

# total_revenue_H2 = np.sum(all_h * price_H2_full * dt)
# total_cost_buy   = np.sum(all_p_buy * price_DA_full * dt / 1000)
# total_rev_sell   = np.sum(all_p_sell * price_DA_full * dt / 1000)
# total_rev_FCR    = np.sum(all_fcr * price_FCR_full * dt_FCR)

# total_profit = total_revenue_H2 - total_cost_buy + total_rev_sell + total_rev_FCR

# print("\n====================")
# print(" FULL SIMULATION PROFIT")
# print("====================")
# print(f"Hydrogen revenue: {total_revenue_H2:,.0f} EUR")
# print(f"Electricity buy cost: {total_cost_buy:,.0f} EUR")
# print(f"Electricity sell revenue: {total_rev_sell:,.0f} EUR")
# print(f"FCR revenue: {total_rev_FCR:,.0f} EUR")
# print("--------------------")
# print(f"TOTAL PROFIT: {total_profit:,.0f} EUR")
# print("====================\n")

# --- CHECK MAX REAL WIND JUMP ---
wind_real_full = np.concatenate([r["wind_real"] for r in results_all])

wind_diff = np.abs(np.diff(wind_real_full))
max_jump = np.max(wind_diff)

print("\n====================")
print(" WIND DATA CHECK")
print("====================")
print(f"Max real wind jump between timesteps: {max_jump:,.0f} kW")
print(f"Mean jump: {np.mean(wind_diff):,.0f} kW")
print("====================\n")


max_forecast_error = np.max(np.abs(forecast_error))

print("====================")
print(" FORECAST ERROR CHECK")
print("====================")
print(f"Max forecast error: {max_forecast_error:,.0f} kW")
print("====================\n")
