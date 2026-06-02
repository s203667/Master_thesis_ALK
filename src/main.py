from kiwisolver import Constraint

from Piecewise import generate_piecewise
import Hydrogen_production as hp
import numpy as np
from Data_preparation import DataPreparationCSV
from Optimizer import InputData, ElectrolyzerModel
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm


USE_STOCHASTIC = False
N_SCENARIOS    = 20
N_NEIGHBORS    = 50

from Scenario_gen import compute_forecast_errors, generate_scenarios
from StochasticOptimizer import StochasticInputData, StochasticElectrolyzerModel

# --- DATA PREPARATION --- #
print("Preparing data...")
prep = DataPreparationCSV(
    start_date="2025-03-05",
    end_date="2026-03-05",
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
    p_out=300,
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
P_min = min(P_stack)      # Minimum power [kW] #CHANGE###
P_max = max(P_stack) # Maximum power [kW] (adjust based on system)
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
num_days =  20#len(wind_real) // TIMESTEPS_PER_DAY

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
        price_sc, wind_sc, imb_sc, probs = generate_scenarios(
            price_fc_day, wind_fc_day, imbalance_fc_day,
            price_errors, wind_errors, imb_errors, wind_forecasts,
            n_scenarios=N_SCENARIOS, n_neighbors = N_NEIGHBORS 
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


# =========================
# CONCATENATE FULL RESULTS
# =========================

P_DA_full       = np.concatenate([d["P_DA"] for d in results_all])
P_grid_full     = np.concatenate([d["P_grid"] for d in results_all])
imbalance_full  = np.concatenate([d["imbalance"] for d in results_all])

h_sell_DA_full  = np.concatenate([d["h_sell_DA"] for d in results_all])
h_sell_RT_full  = np.concatenate([d["h_sell_RT"] for d in results_all])

#P_buy_DA_full   = np.concatenate([d["P_buy_DA"] for d in results_all])
P_buy_RT_full   = np.concatenate([d["P_buy_RT"] for d in results_all])


P_curt_full_RT     = np.concatenate([d["P_curt_RT"] for d in results_all])
P_curt_full_DA     = np.concatenate([d["P_curt_DA"] for d in results_all])

P_elec_RT_full     = np.concatenate([d["P_elec_RT"] for d in results_all])
# Actual and forecast wind for same simulation period
wind_real_full = wind_real[:len(P_DA_full)]
wind_fc_full   = wind_forecast[:len(P_DA_full)]

storage_full = np.concatenate([d["storage"] for d in results_all])

# Forecast error
wind_error_full = wind_real_full - wind_fc_full
total_profit_DA = sum(DA_objectives)
total_profit_RT = sum(RT_objectives)
#total_profit = total_profit_DA + total_profit_RT - sum(d["h_sell_DA"] * price_H2[:len(d["h_sell_DA"])].mean() for d in results_all)

#FCR schedule reconstruction
FCR_schedule_full = np.concatenate([d["FCR_blocks"] for d in results_all])
DA_FCR_full       = np.concatenate([d["DA_FCR"] for d in results_all])
NameDA_FCR_full   = np.concatenate([d["NameDA_FCR"] for d in results_all])
print("RT FCR schedule:", FCR_schedule_full)
print("DA FCR schedule:", DA_FCR_full)
print("NameDA FCR:", NameDA_FCR_full)

for day, d in enumerate(results_all):
    print(f"\nDay {day+1}")
    print("DA FCR:", d["DA_FCR"])
    print("RT FCR:", d["FCR_blocks"])

# Full-horizon series used for the summary and plots
P_elec_DA_full = np.concatenate([d["P_elec_DA"] for d in results_all])
P_elec_RT_full = np.concatenate([d["P_elec_RT"] for d in results_all])
dt = 0.25

print("\n=== FULL HORIZON SUMMARY ===")
print("Total DA objective:", total_profit_DA)
print("Total RT objective:", total_profit_RT)

# Use the full simulation series for realized revenue and imbalance analysis
forecast_error = wind_real_full - wind_fc_full
imbalance_full = P_grid_full - P_DA_full

print("Corr(forecast_error, -imbalance):", np.corrcoef(forecast_error, -imbalance_full)[0, 1])
print("Mean forecast error:", np.mean(forecast_error))
print("Mean imbalance:", np.mean(imbalance_full))
print("Sum forecast error:", np.sum(forecast_error))
print("Sum -imbalance:", -np.sum(imbalance_full))
print("Total timesteps:", len(P_DA_full))
print("Total imbalance:", np.sum(imbalance_full))

print("=== DAY-AHEAD RESULTS ===")
print("P_DA_full first 10:", P_DA_full[:10])
print("Elec load DA first 10:", P_elec_DA_full[:10])
print("Total DA export:", np.sum(P_DA_full))

print("\n=== REAL-TIME RESULTS ===")
print("P_grid_full first 10:", P_grid_full[:10])
print("Elec load RT first 10:", P_elec_RT_full[:10])
print("Total RT export:", np.sum(P_grid_full))

print("\n=== IMBALANCE ===")
print("Imbalance_full first 10:", imbalance_full[:10])
print("Total imbalance:", np.sum(imbalance_full))

# Plot comparisons
plt.figure(figsize=(15,5))
plt.plot(P_DA_full, label="DA export")
plt.plot(P_grid_full, label="RT export")
plt.title("Full-horizon DA vs RT export")
plt.xlabel("Timestep")
plt.ylabel("Power [kW]")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(15,5))
plt.plot(imbalance_full, label="Imbalance")
plt.plot(forecast_error, label="Wind forecast error")
plt.title("Imbalance and wind forecast error")
plt.xlabel("Timestep")
plt.ylabel("kW")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(15,5))
plt.plot(h_sell_DA_full, label="DA H2 sold")
plt.plot(h_sell_RT_full, label="RT H2 sold")
plt.title("Hydrogen sales: DA and RT")
plt.xlabel("Timestep")
plt.ylabel("H2 sold [kg]")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8,5))
plt.scatter(forecast_error, imbalance_full, alpha=0.3)
plt.title("Forecast error vs imbalance")
plt.xlabel("Wind forecast error [kW]")
plt.ylabel("Imbalance [kW]")
plt.grid()
plt.show()

plt.figure(figsize=(10,5))
plt.hist(imbalance_full, bins=100)
plt.title("Imbalance distribution")
plt.xlabel("Imbalance [kW]")
plt.ylabel("Count")
plt.grid()
plt.show()

plt.figure(figsize=(15,5))
plt.plot(wind_real_full, label="Actual wind")
plt.plot(wind_fc_full, label="Forecast wind")
plt.title("Wind forecast vs actual")
plt.xlabel("Timestep")
plt.ylabel("Wind [kW]")
plt.legend()
plt.grid()
plt.show()

# Daily totals
days = np.arange(len(results_all)) + 1
DA_export_daily = np.array([d["P_DA"].sum() for d in results_all]) * dt / 1000
RT_export_daily = np.array([d["P_grid"].sum() for d in results_all]) * dt / 1000
H2_DA_daily = np.array([d["h_sell_DA"].sum() for d in results_all]) * dt
H2_RT_daily = np.array([d["h_sell_RT"].sum() for d in results_all]) * dt
Curtail_daily = np.array([d["P_curt_RT"].sum() for d in results_all]) * dt

plt.figure(figsize=(12,5))
plt.plot(days, DA_export_daily, marker='o', label='DA export (MWh/day)')
plt.plot(days, RT_export_daily, marker='o', label='RT export (MWh/day)')
plt.title('Daily DA and RT export')
plt.xlabel('Day')
plt.ylabel('Energy [MWh/day]')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12,5))
plt.plot(days, H2_DA_daily, marker='o', label='DA H2 sold (kg/day)')
plt.plot(days, H2_RT_daily, marker='o', label='RT H2 sold (kg/day)')
plt.title('Daily hydrogen sales')
plt.xlabel('Day')
plt.ylabel('Hydrogen sold [kg/day]')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12,5))
plt.bar(days, Curtail_daily)
plt.title('Daily RT curtailment energy')
plt.xlabel('Day')
plt.ylabel('Curtailment [MWh/day]')
plt.grid(axis='y')
plt.show()

#Plot a histogram of all imbalance prices
plt.figure(figsize=(10,5))
plt.hist(imbalance_price, bins=50, alpha=0.7)
plt.title("Distribution of imbalance prices")
plt.xlabel("Imbalance price [EUR/MWh]")
plt.ylabel("Count")
plt.grid()
plt.show()


# Revenue decomposition for the full horizon
rev_elec_DA = sum(
    P_DA_full[t] * price_forecast[t] * dt / 1000
    for t in range(len(P_DA_full))
)
rev_h2_DA = sum(
    h_sell_DA_full[t] * price_H2[t] * dt
    for t in range(len(h_sell_DA_full))
)
rev_FCR = sum(
    DA_FCR_full[i] * price_FCR[i] * 16
    for i in range(len(DA_FCR_full))
)
rev_imbalance = -sum(
    imbalance_full[t] * imbalance_price[t] * dt / 1000
    for t in range(len(imbalance_full))
)
cost_buy = sum(
    P_buy_RT_full[t] * price_real[t] * dt / 1000
    for t in range(len(P_buy_RT_full))
)
rev_h2_RT = sum(
    h_sell_RT_full[t] * price_H2[t] * dt
    for t in range(len(h_sell_RT_full))
)

print("\n=== REVENUE BREAKDOWN ===")
print(f"  DA electricity revenue : {rev_elec_DA:,.0f} EUR")
print(f"  DA hydrogen revenue    : {rev_h2_DA:,.0f} EUR")
print(f"  FCR revenue            : {rev_FCR:,.0f} EUR")
print(f"  Imbalance settlement   : {rev_imbalance:,.0f} EUR")
print(f"  Grid purchase cost     : {-cost_buy:,.0f} EUR")
print(f"  RT hydrogen revenue    : {rev_h2_RT:,.0f} EUR")
print(f"  TOTAL                  : {rev_elec_DA + rev_h2_DA + rev_FCR + rev_imbalance - cost_buy + rev_h2_RT:,.0f} EUR")

total_h2_kg = (np.sum(h_sell_DA_full) + np.sum(h_sell_RT_full)) * dt

# Break-even H2 price based on actual realized electricity opportunity cost
opp_cost_elec = sum(
    P_elec_RT_full[t] * price_real[t] * dt / 1000
    for t in range(len(P_elec_RT_full))
)
if total_h2_kg > 0:
    breakeven_h2 = opp_cost_elec / total_h2_kg
    print(f"\n  Opportunity cost of electricity used by electrolyzer: {opp_cost_elec:,.0f} EUR")
    print(f"  Break-even H₂ price    : {breakeven_h2:.2f} EUR/kg")
    print(f"  Current H₂ price       : {price_H2[0]:.2f} EUR/kg")
    print(f"  H₂ margin              : {price_H2[0] - breakeven_h2:+.2f} EUR/kg")

print("="*55 + "\n")




# ── helpers ────────────────────────────────────────────────────────────────
dt = 0.25          # 15-min timesteps → hours
TIMESTEPS_PER_DAY = 96
num_days = len(results_all)

# Re-build arrays that may not exist yet at this point in main.py
P_elec_DA_full = np.concatenate([d["P_elec_DA"] for d in results_all])
P_elec_RT_full = np.concatenate([d["P_elec_RT"] for d in results_all])
P_curt_full    = np.concatenate([d["P_curt_RT"]    for d in results_all])

# Collect z_on from each day's RT model  ← you need to save this in results_all
# Add  "z_on_RT": np.array(RT_model.results.z_on)  to your results_all.append() call.
# Then:
z_on_full = np.concatenate([d["z_on_RT"] for d in results_all])

# FCR per block (one value per 4-h block, 6 blocks/day)
r_FCR_all = []
for d in results_all:
    vals = d.get("r_FCR_DA", None)
    if vals is None:
        # missing key for this day: skip
        continue
    # support dicts (values()), lists or arrays
    if isinstance(vals, dict):
        r_FCR_all.extend(list(vals.values()))
    else:
        try:
            r_FCR_all.extend(list(vals))
        except Exception:
            # non-iterable entry: skip
            continue
r_FCR_all = np.array(r_FCR_all)

# FCR volume caps per block (same ordering as above)
FCR_volume_all = np.concatenate([
    FCR_volume[day * blocks_per_day : (day + 1) * blocks_per_day]
    for day in range(num_days)
])

# ── 1. Electrolyzer utilisation ─────────────────────────────────────────────
total_ts   = len(z_on_full)
on_ts      = int(np.sum(z_on_full))
util_pct   = 100 * on_ts / total_ts

print("\n" + "="*55)
print("  ELECTROLYZER UTILISATION")
print("="*55)
print(f"  Total timesteps      : {total_ts}  ({total_ts * dt:.0f} h)")
print(f"  ON timesteps         : {on_ts}  ({on_ts * dt:.0f} h)")
print(f"  Utilisation rate     : {util_pct:.1f} %")
print(f"  OFF/standby          : {total_ts - on_ts}  ({(total_ts - on_ts) * dt:.0f} h)")

# Daily utilisation (useful to spot low-wind days)
print("\n  Daily utilisation:")
for day in range(num_days):
    day_z = z_on_full[day * TIMESTEPS_PER_DAY : (day + 1) * TIMESTEPS_PER_DAY]
    print(f"    Day {day+1:2d}: {100*day_z.mean():.0f} %")

# ── 2. FCR commitment vs wind availability ──────────────────────────────────
print("\n" + "="*55)
print("  FCR COMMITMENT vs WIND")
print("="*55)

# Expand r_FCR from block level to timestep level
block_size = 16  # 4 h / 15 min
r_FCR_ts = np.repeat(r_FCR_all, block_size)[:len(P_elec_RT_full)]
BLOCK_SIZE = 16  # timesteps per FCR block

# r_FCR_all: shape (num_days * blocks_per_day,)
# We want r_FCR_ts_full: shape (num_days * TIMESTEPS_PER_DAY,)

r_FCR_ts_full = np.repeat(r_FCR_all, BLOCK_SIZE)

# If your total timesteps is not exactly len(r_FCR_all)*BLOCK_SIZE, trim:
r_FCR_ts_full = r_FCR_ts_full[:len(P_elec_RT_full)]
# FCR utilisation
n_blocks       = len(r_FCR_all)
n_active_blks  = int(np.sum(r_FCR_all > 0))
mean_fcr_kw    = r_FCR_all[r_FCR_all > 0].mean() if n_active_blks > 0 else 0
#max_fcr_kw     = r_FCR_all.max()

print(f"  FCR blocks total     : {n_blocks}")
#print(f"  Blocks with FCR > 0  : {n_active_blks}  ({100*n_active_blks/n_blocks:.0f} %)")
print(f"  Mean FCR when active : {mean_fcr_kw:,.0f} kW")
#print(f"  Max FCR              : {max_fcr_kw:,.0f} kW")
print(f"  FCR cap (P_max)      : {P_max:,.0f} kW")

# Feasibility check: at each timestep, does wind cover elec load + FCR headroom?
# The electrolyzer needs to be schedulable at P_elec ± r_FCR.
# Minimum safe wind = P_elec_RT - r_FCR_ts  (downward FCR → elec goes lower, more wind exported)
# Maximum safe wind = P_elec_RT + r_FCR_ts  (upward FCR → elec goes higher, needs more power)
# Red flag: wind < P_elec_RT + r_FCR_ts  AND  no grid buy
wind_real_full = wind_real[:len(P_elec_RT_full)]
# headroom_needed = P_elec_RT_full + r_FCR_ts_full          # power needed for full upward FCR
# headroom_gap    = headroom_needed - (wind_real_full + P_buy_RT_full)
# tight_ts        = int(np.sum(headroom_gap > 1))      # timesteps where upward FCR is tight

# print(f"\n  Upward FCR headroom check:")
# print(f"    Timesteps where wind+buy < elec+FCR : {tight_ts}  "
#       f"({100*tight_ts/len(wind_real_full):.1f} %)")
# print(f"    Max headroom gap  : {headroom_gap.max():,.0f} kW")
# print(f"    Mean headroom gap : {headroom_gap.mean():,.0f} kW  (negative = comfortable)")

# ── 3. Curtailment ──────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  WIND CURTAILMENT")
print("="*55)

total_wind_kWh  = np.sum(wind_real_full) * dt
total_curt_kWh  = np.sum(P_curt_full)   * dt
curt_pct        = 100 * total_curt_kWh / total_wind_kWh if total_wind_kWh > 0 else 0

print(f"  Total wind available : {total_wind_kWh/1e6:.3f} GWh")
print(f"  Total curtailed      : {total_curt_kWh/1e6:.3f} GWh")
print(f"  Curtailment rate     : {curt_pct:.2f} %")
print(f"  Max curtailment (kW) : {P_curt_full.max():,.0f} kW  at t={P_curt_full.argmax()}")

# Curtailment by hour-of-day (spot patterns)
hour_of_day = (np.arange(len(P_curt_full)) % TIMESTEPS_PER_DAY) // 4
curt_by_hour = [P_curt_full[hour_of_day == h].mean() for h in range(24)]
peak_curt_hour = int(np.argmax(curt_by_hour))
print(f"  Peak curtailment hour: {peak_curt_hour:02d}:00  "
      f"(avg {curt_by_hour[peak_curt_hour]:,.0f} kW)")

# Days with >5 % curtailment
print("\n  Days with curtailment > 5 % of available wind:")
for day in range(num_days):
    sl     = slice(day * TIMESTEPS_PER_DAY, (day + 1) * TIMESTEPS_PER_DAY)
    w_day  = np.sum(wind_real_full[sl]) * dt
    c_day  = np.sum(P_curt_full[sl])   * dt
    if w_day > 0 and 100 * c_day / w_day > 5:
        print(f"    Day {day+1:2d}: {100*c_day/w_day:.1f} %  "
              f"({c_day/1e3:.1f} MWh curtailed of {w_day/1e3:.1f} MWh)")

# ── 4. Break-even H₂ price & revenue split ─────────────────────────────────
print("\n" + "="*55)
print("  REVENUE BREAKDOWN & BREAK-EVEN H₂ PRICE")
print("="*55)

# ---- DA stage revenues ----
# Electricity sold to grid (DA)
rev_elec_DA = sum(
    results_all[day]["P_DA"][t] * price_forecast[day * TIMESTEPS_PER_DAY + t] * dt / 1000
    for day in range(num_days)
    for t in range(TIMESTEPS_PER_DAY)
)

# Hydrogen sold (DA stage, valued at H2 price)
rev_h2_DA = sum(
    results_all[day]["h_sell_DA"][t] * price_H2[day * TIMESTEPS_PER_DAY + t] * dt
    for day in range(num_days)
    for t in range(TIMESTEPS_PER_DAY)
)


# FCR revenue (DA bids, earned regardless of RT)
rev_FCR = 0
for day, d in enumerate(results_all):
    r_list = d["DA_FCR"]  # list or array of r_FCR values per block
    for i, r_val in enumerate(r_list):
        block_idx = day * blocks_per_day + i
        # Ensure block_idx does not exceed price_FCR length, but warn if it does
        if block_idx >= len(price_FCR):
            print(f"Warning: block_idx {block_idx} exceeds price_FCR length {len(price_FCR)}. Skipping this block.")
            continue
        rev_FCR += r_val * price_FCR[block_idx] * 16   # dt_FCR = 16 timesteps = 4 h
        rev_FCR += r_val * price_FCR[block_idx] * 16   # dt_FCR = 16 timesteps = 4 h


# ---- RT stage revenues ----
# Imbalance settlement
rev_imbalance = sum(
    (results_all[day]["P_grid"][t] - results_all[day]["P_DA"][t])
    * imbalance_price[day * TIMESTEPS_PER_DAY + t] * dt / 1000
    for day in range(num_days)
    for t in range(TIMESTEPS_PER_DAY)
)

# Grid purchases (cost)
cost_buy = sum(
    results_all[day]["P_buy_RT"][t] * price_real[day * TIMESTEPS_PER_DAY + t] * dt / 1000
    for day in range(num_days)
    for t in range(TIMESTEPS_PER_DAY)
)

# Hydrogen sold (RT stage)
rev_h2_RT = sum(
    results_all[day]["h_sell_RT"][t] * price_H2[day * TIMESTEPS_PER_DAY + t] * dt
    for day in range(num_days)
    for t in range(TIMESTEPS_PER_DAY)
)

total_h2_kg = (np.sum(h_sell_DA_full) + np.sum(h_sell_RT_full)) * dt

print(f"  DA electricity revenue : {rev_elec_DA:>12,.0f} EUR")
print(f"  DA hydrogen revenue    : {rev_h2_DA:>12,.0f} EUR")
print(f"  FCR revenue            : {rev_FCR:>12,.0f} EUR")
print(f"  Imbalance settlement   : {rev_imbalance:>12,.0f} EUR")
print(f"  Grid purchase cost     : {-cost_buy:>12,.0f} EUR")
print(f"  RT hydrogen revenue    : {rev_h2_RT:>12,.0f} EUR")
print(f"  {'─'*35}")
grand_total = rev_elec_DA + rev_h2_DA + rev_FCR + rev_imbalance - cost_buy + rev_h2_RT
print(f"  TOTAL                  : {grand_total:>12,.0f} EUR")

print(f"\n  Total H₂ sold          : {total_h2_kg:>12,.1f} kg")

# Break-even: what H2 price makes H2 revenue = electricity revenue loss from diverting wind
# i.e. opportunity cost of electricity not exported = P_elec_RT * price_DA
opp_cost_elec = sum(
    P_elec_RT_full[day * TIMESTEPS_PER_DAY + t]
    * price_real[day * TIMESTEPS_PER_DAY + t] * dt / 1000
    for day in range(num_days)
    for t in range(TIMESTEPS_PER_DAY)
    if day * TIMESTEPS_PER_DAY + t < len(P_elec_RT_full)
)

if total_h2_kg > 0:
    breakeven_h2 = opp_cost_elec / total_h2_kg
    print(f"\n  Opportunity cost of electricity used by electrolyzer:")
    print(f"    {opp_cost_elec:,.0f} EUR")
    print(f"  Break-even H₂ price    : {breakeven_h2:.2f} EUR/kg")
    print(f"  Current H₂ price       : {price_H2[0]:.2f} EUR/kg")
    margin = price_H2[0] - breakeven_h2
    print(f"  H₂ margin              : {margin:+.2f} EUR/kg  "
          f"({'profitable' if margin > 0 else 'LOSS-MAKING'})")

print("="*55 + "\n")




TIMESTEPS_PER_DAY = 96
num_days = len(results_all)
dt = 0.25  # hours per timestep

# ── rebuild full arrays if not already done ───────────────────────────────
# h_sell_DA_full  = np.concatenate([d["h_sell_DA"]  for d in results_all])
# h_sell_RT_full  = np.concatenate([d["h_sell_RT"]  for d in results_all])
# storage_full    = np.concatenate([d["storage"]    for d in results_all])
# P_buy_RT_full   = np.concatenate([d["P_buy_RT"]   for d in results_all])
# P_elec_RT_full  = np.concatenate([d["P_elec_RT"]  for d in results_all])

T = len(h_sell_DA_full)
hours = np.arange(T) * dt  # x-axis in hours

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 – Hydrogen sales: heatmap (day × hour-of-day)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Hydrogen Sales Pattern  (kg / 15-min slot)", fontsize=14, fontweight="bold")

for ax, data, label in zip(
    axes,
    [h_sell_DA_full, h_sell_RT_full],
    ["DA h_sell", "RT h_sell"],
):
    # Reshape to (days × timesteps_per_day)
    grid = data[: num_days * TIMESTEPS_PER_DAY].reshape(num_days, TIMESTEPS_PER_DAY)

    im = ax.imshow(
        grid,
        aspect="auto",
        origin="upper",
        cmap="YlOrRd",
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, label="kg / slot")

    ax.set_title(label, fontsize=12)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Day")

    # x-ticks every 4 h (every 16 slots)
    tick_slots = np.arange(0, TIMESTEPS_PER_DAY, 16)
    ax.set_xticks(tick_slots)
    ax.set_xticklabels([f"{int(s*0.25):02d}:00" for s in tick_slots])
    ax.set_yticks(np.arange(num_days))
    ax.set_yticklabels([f"Day {d+1}" for d in range(num_days)], fontsize=8)

    # Overlay total-sold-per-day as right-side annotation
    for d in range(num_days):
        total = grid[d].sum()
        ax.text(
            TIMESTEPS_PER_DAY + 0.5, d, f"{total:,.0f} kg",
            va="center", ha="left", fontsize=7, color="#555"
        )

plt.tight_layout(rect=[0, 0, 0.92, 1])
#plt.savefig("h_sell_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
#print("Saved: h_sell_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 – DA vs RT h_sell per day (stacked bar)
# Shows whether RT is systematically dominating DA sales
# ─────────────────────────────────────────────────────────────────────────────
da_daily  = h_sell_DA_full[:num_days*TIMESTEPS_PER_DAY].reshape(num_days, TIMESTEPS_PER_DAY).sum(axis=1)
rt_daily  = h_sell_RT_full[:num_days*TIMESTEPS_PER_DAY].reshape(num_days, TIMESTEPS_PER_DAY).sum(axis=1)
days_idx  = np.arange(num_days)

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(days_idx, da_daily, label="DA h_sell", color="#2196F3", alpha=0.85)
ax.bar(days_idx, rt_daily, bottom=da_daily, label="RT h_sell", color="#FF5722", alpha=0.85)
ax.set_xticks(days_idx)
ax.set_xticklabels([f"D{d+1}" for d in days_idx])
ax.set_ylabel("Total H₂ sold (kg)")
ax.set_title("Daily hydrogen sales: DA vs RT split", fontweight="bold")
ax.legend()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
plt.tight_layout()
#plt.savefig("h_sell_daily_split.png", dpi=150)
plt.show()
#print("Saved: h_sell_daily_split.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 – P_buy vs Storage level (the key operational relationship)
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10))
gs  = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.35)

# ── top panel: storage + P_buy over full horizon ──────────────────────────
ax1 = fig.add_subplot(gs[0])
ax2 = ax1.twinx()

hours_full = np.arange(T) * dt

ln1 = ax1.fill_between(
    hours_full, storage_full,
    alpha=0.35, color="#1565C0", label="H₂ storage (kg)"
)
ln1b = ax1.plot(hours_full, storage_full, color="#1565C0", lw=1, alpha=0.7)

# P_buy – highlight buying episodes in red
P_buy_pos = np.where(P_buy_RT_full > 0, P_buy_RT_full, 0)
ln2 = ax2.fill_between(
    hours_full, P_buy_pos,
    alpha=0.6, color="#C62828", label="P_buy RT (kW)"
)
ln2b = ax2.plot(hours_full, P_buy_pos, color="#C62828", lw=0.8, alpha=0.8)

# Day separators
storage_max = storage_full.max()
for d in range(1, num_days):
    ax1.axvline(d * TIMESTEPS_PER_DAY * dt, color="#888", lw=0.6, ls="--", alpha=0.5)
    ax1.text(
        d * TIMESTEPS_PER_DAY * dt + 0.5, storage_max * 0.97,
        f"D{d+1}", fontsize=7, color="#666"
    )

ax1.set_ylabel("H₂ Storage level (kg)", color="#1565C0")
ax2.set_ylabel("P_buy RT (kW)", color="#C62828")
ax1.tick_params(axis="y", colors="#1565C0")
ax2.tick_params(axis="y", colors="#C62828")
ax1.set_xlabel("Time (hours)")
ax1.set_title("Storage level vs Grid purchases (RT)", fontweight="bold", fontsize=12)

lines = [
    plt.Line2D([0], [0], color="#1565C0", lw=2, label="H₂ storage"),
    plt.Line2D([0], [0], color="#C62828", lw=2, label="P_buy RT"),
]
ax1.legend(handles=lines, loc="upper right")

# ── bottom panel: scatter – does P_buy happen when storage is low? ────────
ax3 = fig.add_subplot(gs[1])

# colour by electrolyzer on/off
on_mask  = P_elec_RT_full > 0
off_mask = ~on_mask

sc_on  = ax3.scatter(
    storage_full[on_mask & (P_buy_RT_full > 0)],
    P_buy_RT_full[on_mask & (P_buy_RT_full > 0)],
    s=10, alpha=0.5, color="#E53935", label="Elec ON + buying"
)
sc_off = ax3.scatter(
    storage_full[off_mask & (P_buy_RT_full > 0)],
    P_buy_RT_full[off_mask & (P_buy_RT_full > 0)],
    s=10, alpha=0.5, color="#FB8C00", label="Elec OFF + buying"
)

ax3.set_xlabel("H₂ Storage level (kg)")
ax3.set_ylabel("P_buy RT (kW)")
ax3.set_title(
    "When does P_buy happen? (storage level vs grid purchase size)",
    fontweight="bold"
)
ax3.legend(fontsize=9)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

# Add correlation annotation
buying_mask = P_buy_RT_full > 0
if buying_mask.sum() > 2:
    corr = np.corrcoef(storage_full[buying_mask], P_buy_RT_full[buying_mask])[0, 1]
    ax3.text(
        0.02, 0.92,
        f"Corr(storage, P_buy | buying>0) = {corr:.3f}",
        transform=ax3.transAxes, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
    )

n_buy_ts  = int(buying_mask.sum())
buy_pct   = 100 * n_buy_ts / T
total_buy = P_buy_RT_full.sum() * dt / 1000  # MWh
print(f"\n── P_buy summary ────────────────────────────────")
print(f"  Timesteps with P_buy > 0 : {n_buy_ts} / {T}  ({buy_pct:.1f} %)")
print(f"  Total grid energy bought  : {total_buy:,.1f} MWh")
print(f"  Max single purchase       : {P_buy_RT_full.max():,.0f} kW")
print(f"  Mean purchase (when > 0)  : {P_buy_RT_full[buying_mask].mean():,.0f} kW")

if buying_mask.sum() > 2:
    print(f"  Corr(storage, P_buy)      : {corr:.3f}")
    if corr < -0.3:
        print("  → P_buy tends to occur when storage is LOW  ✓ (expected behaviour)")
    elif corr > 0.3:
        print("  → WARNING: P_buy tends to occur when storage is HIGH  ✗")
    else:
        print("  → P_buy is largely independent of storage level (check model)")

plt.tight_layout()
#plt.savefig("pbuy_vs_storage.png", dpi=150, bbox_inches="tight")
plt.show()
#print("Saved: pbuy_vs_storage.png")







TIMESTEPS_PER_DAY = 96
dt = 0.25

P_buy_RT_full   = np.concatenate([d["P_buy_RT"]  for d in results_all])
P_curt_RT_full  = np.concatenate([d["P_curt_RT"] for d in results_all])
P_grid_full     = np.concatenate([d["P_grid"]    for d in results_all])
P_DA_full       = np.concatenate([d["P_DA"]      for d in results_all])
P_elec_RT_full  = np.concatenate([d["P_elec_RT"] for d in results_all])
wind_RT_full    = np.concatenate([d["wind_real"]   for d in results_all])  # actual wind
storage_full    = np.concatenate([d["storage"]   for d in results_all])

T = len(P_buy_RT_full)

# ── 1. Simultaneous curtailment AND buying ───────────────────────────────
both_mask = (P_buy_RT_full > 1) & (P_curt_RT_full > 1)
n_both    = both_mask.sum()
pct_both  = 100 * n_both / T

print("\n── Simultaneous P_buy + curtailment ─────────────────────────")
print(f"  Timesteps with BOTH P_buy>0 and P_curt>0 : {n_both} / {T}  ({pct_both:.1f} %)")
if n_both > 0:
    print(f"  Mean P_buy   during overlap : {P_buy_RT_full[both_mask].mean():,.0f} kW")
    print(f"  Mean P_curt  during overlap : {P_curt_RT_full[both_mask].mean():,.0f} kW")
    print(f"  → This is economically irrational and indicates a model bug.")

# ── 2. Wind balance check ────────────────────────────────────────────────
# wind + P_buy should == P_elec + P_curt + P_grid  (ignoring standby which is small)
lhs = wind_RT_full + P_buy_RT_full
rhs = P_elec_RT_full + P_curt_RT_full + P_grid_full
residual = lhs - rhs
print("\n── Wind balance residual (should be ~0) ──────────────────────")
print(f"  Max  |residual| : {np.abs(residual).max():,.1f} kW")
print(f"  Mean |residual| : {np.abs(residual).mean():,.2f} kW")
if np.abs(residual).max() > 10:
    print("  !! Large residuals — standby load or other term may be missing")

# ── 3. Curtailment breakdown by cause ───────────────────────────────────
print("\n── When does curtailment happen? ─────────────────────────────")

# A: electrolyzer at P_max
elec_at_max = P_elec_RT_full > (0.98 * P_elec_RT_full.max())
curt_at_elec_max = (P_curt_RT_full > 1) & elec_at_max
print(f"  Elec at P_max AND curtailing     : {curt_at_elec_max.sum()} ts")

# B: P_grid at grid_cap (need to know grid_cap — adjust if different)
grid_cap = P_grid_full.max()  # inferred; replace with actual value
curt_at_grid_cap = (P_curt_RT_full > 1) & (P_grid_full > 0.98 * grid_cap)
print(f"  P_grid near cap AND curtailing   : {curt_at_grid_cap.sum()} ts  (inferred cap={grid_cap:,.0f} kW)")

# C: P_grid constrained by proximity to P_DA (deviation limit?)
deviation = P_grid_full - P_DA_full
max_pos_dev = deviation.max()
print(f"\n  Max positive deviation P_grid-P_DA : {max_pos_dev:,.0f} kW")
print(f"  Max negative deviation P_grid-P_DA : {deviation.min():,.0f} kW")
print(f"  Mean deviation                     : {deviation.mean():,.0f} kW")
# If max positive deviation is suspiciously round (e.g. exactly 10000, 50000),
# it suggests a hard deviation cap is binding
curt_with_large_dev = (P_curt_RT_full > 1) & (deviation > 0.95 * max_pos_dev)
print(f"  Curtailment timesteps where deviation near max : {curt_with_large_dev.sum()}")

# ── 4. Day 1 deep dive (worst curtailment daSy) ──────────────────────────
print("\n── Day 2 detail ────────────────────────────")
sl = slice(1, TIMESTEPS_PER_DAY)
for label, arr in [
    ("wind_RT  ", wind_RT_full[sl]),
    ("P_elec_RT   ", P_elec_RT_full[sl]),
    ("P_elec_DA   ", P_elec_DA_full[sl]),
    ("P_grid   ", P_grid_full[sl]),
    ("P_DA     ", P_DA_full[sl]),
    ("P_buy    ", P_buy_RT_full[sl]),
    ("P_curt   ", P_curt_RT_full[sl]),
    ("storage  ", storage_full[sl]),
]:
    print(f"  {label}  min={arr.min():>10,.0f}  max={arr.max():>10,.0f}  mean={arr.mean():>10,.0f}")

# ── 5. P_buy when storage is high ───────────────────────────────────────
storage_pct75 = np.percentile(storage_full, 75)
buy_when_full = (P_buy_RT_full > 1) & (storage_full > storage_pct75)
print(f"\n── P_buy when storage is in top 25% (>{storage_pct75:,.0f} kg) ──────")
print(f"  {buy_when_full.sum()} timesteps  ({100*buy_when_full.sum()/T:.1f} %)")
print(f"  Mean P_buy in those slots : {P_buy_RT_full[buy_when_full].mean():,.0f} kW")
if buy_when_full.sum() > 50:
    print("  !! Buying grid power when storage is full — irrational, check P_buy cost term")

##################################
#NY###
# === CORRECT REVENUE BREAKDOWN FOR STOCHASTIC RUN ===

# 1. DA electricity revenue: P_DA committed, cleared at actual DA price
rev_elec_DA = sum(
    P_DA_full[t] * price_real[t] * dt / 1000   # ← price_real, not price_forecast
    for t in range(len(P_DA_full))
)

# 2. Imbalance settlement: actual deviation at actual imbalance price
rev_imbalance = sum(
    (P_grid_full[t] - P_DA_full[t]) * imbalance_price[t] * dt / 1000
    for t in range(len(P_grid_full))
)

# 3. RT hydrogen revenue: only real sales, from RT model
rev_h2_RT = sum(
    h_sell_RT_full[t] * price_H2[t] * dt
    for t in range(len(h_sell_RT_full))
)

# 4. Grid purchase cost: from RT model
cost_buy = sum(
    P_buy_RT_full[t] * price_real[t] * dt / 1000
    for t in range(len(P_buy_RT_full))
)

# 5. FCR revenue: committed in DA, earned regardless
rev_FCR = sum(
    DA_FCR_full[i] * price_FCR[i] * 4   # dt_FCR = 4 hours, not 16 timesteps
    for i in range(len(DA_FCR_full))
)

# DA hydrogen revenue: DO NOT include — it is a scenario planning artifact
# h_sell_DA is from the stochastic Stage 2, not a real physical transaction

total = rev_elec_DA + rev_imbalance + rev_h2_RT - cost_buy + rev_FCR

print("\n=== REALIZED REVENUE BREAKDOWN (Stochastic DA + RT) ===")
print(f"  DA electricity revenue : {rev_elec_DA:,.0f} EUR")
print(f"  Imbalance settlement   : {rev_imbalance:,.0f} EUR")
print(f"  RT hydrogen revenue    : {rev_h2_RT:,.0f} EUR")
print(f"  Grid purchase cost     : {-cost_buy:,.0f} EUR")
print(f"  FCR revenue            : {rev_FCR:,.0f} EUR")
print(f"  ─────────────────────────────────────")
print(f"  TOTAL REALIZED         : {total:,.0f} EUR")

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