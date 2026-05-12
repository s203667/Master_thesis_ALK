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
    end_date="2025-04-05",
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
#H = 96  # e.g. 1 day (96 × 15min)
S = len(A_s)      # Number of segments
I = len(fcr_df)   # Number of FCR blocks (4-hour intervals)


# Data
wind_real = wind_df['wind_kW'].values  # Wind power [kW]
price_real = el_price_df['el_price[EUR/MWh]'].values  # Day-ahead price [EUR/MWh]
price_H2 = np.full(len(wind_real), 5.5)
#price_H2 = np.full(H, 5.5)  # Hydrogen price [EUR/kg], constant for now, from European Hydrogen Observatory
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
P_sb = min(P_stack)     # Standby power [kW] #CHANGE###

#Add "Forecast" that is just the day before repeated
lag = 96  # 1 day

 #FCR is daily blocks, not per timestep

wind_forecast = np.roll(wind_real, lag)
price_forecast = np.roll(price_real, lag)

# For the first day, fill with actual values (or zeros)
wind_forecast = np.roll(wind_real, lag)
price_forecast = np.roll(price_real, lag)
wind_forecast[:lag] = wind_real[:lag]
price_forecast[:lag] = price_real[:lag]

# print("NaNs in FCR price:", np.isnan(price_FCR).sum())
# print("NaNs in FCR volume:", np.isnan(FCR_volume).sum())

#r_FCR_block = 0  ####TEMPORARILY WHILE TESTING #####
# Storage parameters
storage_max = 2000.0     # Max storage [kg]
storage_initial = 0.0     # Initial storage [kg]
N_down = 3             # Downtime in 15min intervals, adjust as needed

t0 = 0

storage_state = storage_initial
z_on_state = 1
P_prev = 0

TIMESTEPS_PER_DAY = 96
num_days =  20 #len(wind_real) // TIMESTEPS_PER_DAY

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
    # price_FCR_day = price_FCR[day:day+1]
    # FCR_volume_day = FCR_volume[day:day+1] #
    print(f"\n=== DAY {day+1} INITIAL STATE ===")
    print("storage_state:", storage_state)
    print("z_on_state:", z_on_state)
    print("P_prev:", P_prev)

    # --- 4. Build model ---
    input_DA = InputData(
        T=TIMESTEPS_PER_DAY,
        S=S,
        I=len(price_FCR_day),  # number of FCR blocks
        wind=None,
        wind_forecast=wind_fc_day,
        price_DA=price_fc_day,
        price_H2=price_H2_day,
        price_FCR=price_FCR_day,
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
        grid_cap=grid_cap,
        N_down=N_down,
        FCR_volume=FCR_volume,
        z_on_prev=z_on_state,
        P_prev=P_prev,
        is_DA=True,
        DA_export=None  # Not used in DA model
    )

    DA_model = ElectrolyzerModel(input_DA)
    # print("Running DA for day", day)
    print("storage_state:", storage_state)
    print("z_on_state:", z_on_state)
    print("P_prev:", P_prev)
    DA_model.run()
    print(f"\n--- Window starting at t0={t0} ---")
    print(f"  storage_initial passed to model: {storage_state}")

    P_DA = np.array(DA_model.results.P_DA) # DA export schedule
    P_Elec_DA = np.array([
        sum(DA_model.model.P_seg[t, s].value for s in DA_model.model.S)
        + DA_model.model.P_sb[t].value
        for t in DA_model.model.T
    ])
    h_sell_DA = np.array(DA_model.results.h_sell)
    P_buy_DA = np.array(DA_model.results.P_buy)
    DA_objectives.append(DA_model.results.objective_value)

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
        I=len(price_FCR),
        wind=wind_rt_day,             # actual wind
        wind_forecast=None,
        price_DA=price_rt_day,        # actual DA price
        price_H2=price_H2_RT,
        price_FCR=price_FCR,
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
        grid_cap=grid_cap,
        N_down=N_down,
        FCR_volume=FCR_volume,
        z_on_prev=z_on_state,
        P_prev=P_prev,
        is_DA=False,
        DA_export=P_DA             # fixed DA bid
    )


    RT_model = ElectrolyzerModel(input_RT)
    RT_model.run()

    P_grid    = np.array(RT_model.results.P_grid)
    h_sell_RT = np.array(RT_model.results.h_sell)
    P_buy_RT  = np.array(RT_model.results.P_buy)

    RT_objectives.append(RT_model.results.objective_value)
    
    #Update states for next day
    storage_state = RT_model.results.storage[-1]
    z_on_state = RT_model.results.z_on[-1]
    P_prev = RT_model.results.P_elec[-1]

    print(f"\nDAY {day+1}")
    print("  FCR blocks this day:", price_FCR_day)
    print("  len(price_FCR_day):", len(price_FCR_day))
    print("  I passed to model:", len(price_FCR_day))

    #Save results
    results_all.append({
        "P_DA": P_DA,
        "P_grid": P_grid,
        "h_sell_DA": h_sell_DA,
        "h_sell_RT": h_sell_RT,
        "P_buy_DA": P_buy_DA,
        "P_buy_RT": P_buy_RT,
        "imbalance": np.array(P_DA) - np.array(P_grid),
        "storage_end": storage_state,
        "Total profit": DA_model.results.objective_value + RT_model.results.objective_value - sum(h_sell_DA * price_H2_day)
    })

# results_all.append({
#     "h": model.results.h[:step_eff],
#     "storage": model.results.storage[:step_eff],
#     "z_on": model.results.z_on[:step_eff],
#     "P_buy": model.results.P_buy[:step_eff],
#     "P_DA": model.results.P_DA[:step_eff],
#     "P_curt": model.results.P_curt[:step_eff],
#     "r_FCR_block": model.results.r_FCR_block[:step_eff],
#     "wind_real": wind_slice_real,
#     "price_real": price_slice_real,
#     "wind_forecast": wind_slice_forecast[:step_eff],
#     "price_forecast": price_slice_forecast[:step_eff],
# })
# else:
#     print(f"Warning: Model infeasible at window starting at t0={t0}, skipping this window.")
#     break
#     # Optionally, handle infeasible cases (e.g., set default values or break)


P_DA_full = np.concatenate([d["P_DA"] for d in results_all])
P_grid_full = np.concatenate([d["P_grid"] for d in results_all])
imbalance_full = np.concatenate([d["imbalance"] for d in results_all])

total_profit_DA = sum(DA_objectives)
total_profit_RT = sum(RT_objectives)
total_profit = total_profit_DA + total_profit_RT - sum(d["h_sell_DA"] * price_H2[:len(d["h_sell_DA"])].mean() for d in results_all)

# full series
wind_real_full     = wind_real[:len(P_DA_full)]
wind_forecast_full = wind_forecast[:len(P_DA_full)]

forecast_error = wind_real_full - wind_forecast_full      # e_wind[t]
imbalance_full = P_DA_full - P_grid_full                  # your sign

print("Corr(forecast_error, -imbalance):",
      np.corrcoef(forecast_error, -imbalance_full)[0, 1])

print("Mean forecast error:", np.mean(forecast_error))
print("Mean imbalance:", np.mean(imbalance_full))

print("Sum forecast error:", np.sum(forecast_error))
print("Sum -imbalance:", -np.sum(imbalance_full))


print("Total timesteps:", len(P_DA_full))
print("Total imbalance:", np.sum(imbalance_full))

print("=== DAY-AHEAD RESULTS ===")
print("P_DA first 10:", P_DA[:10])
print("Elec load DA first 10:", P_Elec_DA[:10])
print("Total DA export:", np.sum(P_DA))

print("\n=== REAL-TIME RESULTS ===")
P_grid = RT_model.results.P_grid[:]
P_elec_RT = RT_model.results.P_elec[:]

print("P_grid first 10:", P_grid[:10])
print("Elec load RT first 10:", P_elec_RT[:10])
print("Total RT export:", np.sum(P_grid))
 
print("\n=== IMBALANCE ===")
imbalance = np.array(P_DA) - np.array(P_grid)
print("Imbalance first 10:", imbalance[:10])
print("Total imbalance:", np.sum(imbalance))

# wind_fc = wind_forecast[:H]
# wind_rt = wind_real[:H]

# wind_error = wind_rt - wind_fc

# print("Max wind error:", np.max(np.abs(wind_error)))
# #print total wind error
# print("Total wind error:", np.sum(wind_error))

# # Access results
# print("Hydrogen production:", [h for result in results_all for h in result["h"][:5]])
#   # First 5 values
# print("Storage levels:", [storage for result in results_all for storage in result["storage"][:5]])
# print("wind power last ten periods:", wind[:10])
# print("Maximum wind power:", np.max(wind))
# print("Final t0:", t0)
# print("Expected t0:", len(wind))

# print(f"price_FCR_slice length: {len(price_FCR_slice)}")
# print(f"FCR_volume_slice length: {len(FCR_volume_slice)}")
# # Debug: Check actual z_on values
# print("\n--- DEBUG ---")
# print(f"Number of results: {len(results_all)}")
# if results_all:
#     print(f"First result z_on values (first 20):", results_all[0]["z_on"][:20])
#     print(f"First result z_on unique values:", set(results_all[0]["z_on"]))
#     print(f"First result z_on dtype: {type(results_all[0]['z_on'][0])}")

# def reconstruct_full_timeseries(results_all, step):
#     all_h = []
#     all_storage = []
#     all_z_on = []
#     all_p_buy = []
#     all_P_DA = []
#     all_fcr_blocks = []

#     for i, result in enumerate(results_all):
#         # Determine how many steps this window contributes
#         if i == len(results_all) - 1:
#             H_res = len(result["h"])   # last window: take all
#         else:
#             H_res = step            # earlier windows: take only first `step`

#         # Append exactly the used part
#         all_h.extend(result["h"][:H_res])
#         all_storage.extend(result["storage"][:H_res])
#         all_z_on.extend(result["z_on"][:H_res])
#         all_p_buy.extend(result["P_buy"][:H_res])
#         all_P_DA.extend(result["P_DA"][:H_res])

#         # Expand FCR blocks to 16 timesteps each
#         for block_value in result["r_FCR_block"]:
#             all_fcr_blocks.extend([block_value] * 16)

#     return (
#         np.array(all_h),
#         np.array(all_storage),
#         np.array(all_z_on),
#         np.array(all_p_buy),
#         np.array(all_P_DA),
#         np.array(all_fcr_blocks),
#     )

# def downsample(arr, factor=20):
#     return arr[::factor]


# # --- RECONSTRUCT FULL YEAR ---
# all_h, all_storage, all_z_on, all_p_buy, all_P_DA, all_fcr = \
#     reconstruct_full_timeseries(results_all, step)

# h_roll = all_h.copy()
# print(f"Full timeline length: {len(all_h)} timesteps")

# # --- DEBUG CHECK ---
# print("First 20 z_on:", all_z_on[:20])
# print("Unique z_on values:", set(all_z_on[:500]))

# # --- SUMMARY ---
# print("\nFirst 500 timesteps summary:")
# print(f"Total H2 production: {sum(all_h[:500]):.1f} kg")
# print(f"On periods: {sum(all_z_on[:500])} out of 500")

# # --- PLOTS (first 500 timesteps) ---
# plt.figure(figsize=(12,8))

# plt.subplot(3,1,1)
# plt.plot(all_z_on, linewidth=0.8)
# plt.title("Electrolyzer On/Off (first 500 timesteps)")
# plt.grid(True, alpha=0.3)

# plt.subplot(3,1,2)
# plt.plot(all_h, linewidth=0.8)
# plt.title("H2 Production (kg/h)")
# plt.grid(True, alpha=0.3)

# plt.subplot(3,1,3)
# plt.plot(all_storage, linewidth=0.8)
# plt.title("Storage Level (kg)")
# plt.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()

# # --- FULL-YEAR PLOTS (DOWNSAMPLED) ---
# plt.figure(figsize=(14,10))

# # plt.subplot(4,1,1)
# # plt.plot(downsample(all_z_on, 20))
# # plt.title("Electrolyzer On/Off (downsampled)")
# # plt.grid(True, alpha=0.3)

# plt.subplot(4,1,1)
# plt.plot(downsample(all_h, 20))
# plt.plot(downsample(all_storage, 20))
# plt.legend(["H2 Production", "Storage Level"])
# plt.title("H2 Production (kg/h, downsampled)")
# plt.grid(True, alpha=0.3)

# plt.subplot(4,1,2)
# plt.plot(downsample(all_p_buy, 20), label="Buy")
# plt.plot(downsample(all_P_DA, 20), label="Sell")
# plt.legend()
# plt.title("Electricity Trading (downsampled)")
# plt.grid(True, alpha=0.3)

# plt.subplot(4,1,3)
# plt.plot(downsample(all_fcr, 20))
# plt.title("FCR Power (downsampled)")
# plt.grid(True, alpha=0.3)


# plt.tight_layout()
# plt.show()

# wind_real_full = np.concatenate([r["wind_real"] for r in results_all])
# wind_forecast_full = np.concatenate([r["wind_forecast"] for r in results_all])

# forecast_error = wind_real_full - wind_forecast_full

# plt.plot(forecast_error)
# plt.title("Wind forecast error")
# plt.xlabel("Timestep")
# plt.ylabel("Forecast error (kW)")
# plt.grid(True, alpha=0.3)
# plt.show()

# # --- COMPUTE FULL-SIMULATION OBJECTIVE ---
# dt = 0.25          # 15 minutes
# dt_FCR = 4         # FCR blocks are 4 hours

# # Slice price arrays to match lengths
# price_H2_full = price_H2[:len(all_h)]
# price_DA_full = price_DA[:len(all_h)]

# # FCR price must match number of blocks
# price_FCR_full = price_FCR[:len(all_fcr)]

# total_revenue_H2 = np.sum(all_h * price_H2_full * dt)
# total_cost_buy   = np.sum(all_p_buy * price_DA_full * dt / 1000)
# total_rev_sell   = np.sum(all_P_DA * price_DA_full * dt / 1000)
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
# wind_real_full = np.concatenate([r["wind_real"] for r in results_all])

# wind_diff = np.abs(np.diff(wind_real_full))
# max_jump = np.max(wind_diff)

# print("\n====================")
# print(" WIND DATA CHECK")
# print("====================")
# print(f"Max real wind jump between timesteps: {max_jump:,.0f} kW")
# print(f"Mean jump: {np.mean(wind_diff):,.0f} kW")
# print("====================\n")


# max_forecast_error = np.max(np.abs(forecast_error))

# print("====================")
# print(" FORECAST ERROR CHECK")
# print("====================")
# print(f"Max forecast error: {max_forecast_error:,.0f} kW")
# print("====================\n")



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