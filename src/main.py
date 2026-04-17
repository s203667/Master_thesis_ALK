from Piecewise import generate_piecewise
import Hydrogen_production as hp
import numpy as np
from Data_preparation import DataPreparationCSV
from Optimizer import InputData, ElectrolyzerModel
import matplotlib.pyplot as plt

# --- DATA PREPARATION --- #
print("Preparing data...")
prep = DataPreparationCSV(
    start_date="2016-01-01",
    end_date="2016-12-31",  # Shorter period for testing
    el_price_file_name="",
    wind_file_name=""
)
data = prep.data

wind_df = data['wind']
el_price_df = data['el_price']
fcr_df = data['fcr']

print("Data prepared.")

# --- PIECEWISE APPROXIMATION --- #
print("Generating piecewise approximation...")
A_cell = 0.0625 # m²
j_mA_cm2 = np.linspace(1, 400, 50)
j = j_mA_cm2 * 10          # A/m²
I_range = j * A_cell

A_s, B_s, P_seg_min, P_seg_max, P_total, m_dot = generate_piecewise(
    hp,
    T=80,
    I_range=I_range,
    A_cell=0.0625,
    p=40,
    p_out=45,
    N_stacks=40,
    N_cell=250,
    n_segments=6,
    plot=False  # Set to True to see plot
)

print("Piecewise approximation generated.")

# --- OPTIMIZATION SETUP --- #
print("Setting up optimization...")

# Time parameters
T = len(wind_df)  # Number of time steps (15-min intervals)
S = len(A_s)      # Number of segments
I = len(fcr_df)   # Number of FCR blocks (4-hour intervals)

# Data
wind = wind_df['wtc_ActPower_mean'].values  # Wind power [kW]
price_DA = el_price_df['el_price[EUR/MWh]'].values  # Day-ahead price [EUR/MWh]
price_H2 = np.full(T, 5.5)  # Hydrogen price [EUR/kg], constant for now, from European Hydrogen Observatory
price_FCR = fcr_df['price_FCR'].values  # FCR price [EUR/MW]
FCR_volume = fcr_df["volume_up"].values * 1000 #consider not multiplying by 1000? #Max volume possible to bid

# Electrolyzer parameters
P_min = 0.0      # Minimum power [kW]
P_max = 4000.0  # Maximum power [kW] (adjust based on system)
P_sb = 100.0     # Standby power [kW]

# Storage parameters
storage_max = 1000.0     # Max storage [kg]
storage_initial = 0.0     # Initial storage [kg]
N_down = 3             # Downtime in 15min intervals, adjust as needed

# Create input data
input_data = InputData(
    T=T, S=S, I=I,
    wind=wind,
    price_DA=price_DA,
    price_H2=price_H2,
    price_FCR=price_FCR,
    P_min=P_min, P_max=P_max, P_sb=P_sb,
    P_seg_min=P_seg_min, P_seg_max=P_seg_max,
    A_s=A_s, B_s=B_s,
    storage_max=storage_max,
    storage_initial=storage_initial,
    N_down=N_down, 
    FCR_volume=FCR_volume
)

# Create and solve model
model = ElectrolyzerModel(input_data)
print("Optimization model created.")

# Solve
model.run()
print("Optimization completed.")

# Access results
print("Hydrogen production:", model.results.h[:5])  # First 5 values
print("Storage levels:", model.results.storage[:5])
print(wind[:10])
print(np.max(wind))

# Analyze results
# Plot H2 production and storage
plt.figure(figsize=(12,6))
plt.subplot(2,1,1)
plt.plot(model.results.h[:100])  # First 100 periods
plt.title('H2 Production (kg/h)')
plt.subplot(2,1,2)
plt.plot(model.results.storage[:100])
plt.title('Storage Level (kg)')
plt.show()

# Check operation stats
# Check operation stats
on_periods = sum(model.results.z_on)
total_h2 = sum(model.results.h)
print(f"Electrolyzer on for {on_periods} periods")
print(f"Total H2 produced: {total_h2:.1f} kg")

# Plot more
plt.figure(figsize=(12,8))
plt.subplot(3,1,1)
plt.plot(model.results.z_on[:100])
plt.title('Electrolyzer On/Off')
plt.subplot(3,1,2)
plt.plot(model.results.P_buy[:100], label='Buy')
plt.plot(model.results.P_sell[:100], label='Sell')
plt.legend()
plt.title('Electricity Trading (kW)')
# Expand FCR block values to per-time-step
fcr_per_t = []
for i in range(len(model.results.r_FCR_block)):
    fcr_value = model.results.r_FCR_block[i]
    fcr_per_t.extend([fcr_value] * 16)  # 16 × 15-min = 4 hours

plt.subplot(3,1,3)
plt.plot(fcr_per_t[:100])  # Now per time step
plt.title('FCR Power (kW)')
plt.tight_layout()
plt.show()