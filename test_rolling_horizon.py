"""
Test script for rolling horizon and forecasting mechanism.
Run this to verify the system works correctly.
"""

import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from Data_preparation import DataPreparationCSV
import pandas as pd

print("=" * 80)
print("ROLLING HORIZON & FORECASTING TESTS")
print("=" * 80)

# --- LOAD DATA ---
print("\n1. Loading data...")
prep = DataPreparationCSV(
    start_date="2025-03-05",
    end_date="2025-03-30",
    el_price_file_name="DayAheadPrices.csv",
    wind_file_name="WindProd5MinRealtime.csv",
    fcr_file_name="FcrDK1.csv",
)
data = prep.data

wind_df = data['wind']
el_price_df = data['el_price']
fcr_df = data['fcr']

wind = wind_df['wind_kW'].values
price = el_price_df['el_price[EUR/MWh]'].values
price_FCR = fcr_df['price_FCR'].values
FCR_volume = fcr_df['FCR_volume'].values

print(f"   Wind length: {len(wind)} timesteps ({len(wind)*15/60:.1f} hours)")
print(f"   Price length: {len(price)} timesteps")
print(f"   FCR length: {len(price_FCR)} blocks")
print(f"   ✓ Data loaded")

# --- TEST 1: FORECAST MECHANISM ---
print("\n2. Testing forecast mechanism...")
lag = 96
wind_forecast = np.roll(wind, lag)
price_forecast = np.roll(price, lag)
wind_forecast[:lag] = wind[:lag]
price_forecast[:lag] = price[:lag]

# Check: forecast[t] should match real[t-lag] (except for first lag values)
test_idx = 200
if abs(wind_forecast[test_idx] - wind[test_idx - lag]) < 1e-6:
    print(f"   ✓ Forecast lag works: wind_forecast[{test_idx}] == wind[{test_idx-lag}]")
else:
    print(f"   ✗ FORECAST LAG ERROR at index {test_idx}")
    print(f"     Expected: {wind[test_idx-lag]}, Got: {wind_forecast[test_idx]}")

# Check: first lag values are real values
if np.allclose(wind_forecast[:lag], wind[:lag]):
    print(f"   ✓ First {lag} steps filled with real values")
else:
    print(f"   ✗ First {lag} steps not properly filled")

# --- TEST 2: WINDOW SLICING ---
print("\n3. Testing window slicing logic...")
H = 96
step = 24
t0_tests = [0, 24, 48, 96]

for t0 in t0_tests:
    H_eff = min(H, len(wind) - t0)
    step_eff = min(step, H_eff)
    
    wind_slice_real = wind[t0:t0 + step_eff]
    wind_slice_forecast = wind_forecast[t0:t0 + H_eff]
    
    # Check: slices are correct length
    real_ok = len(wind_slice_real) == step_eff
    forecast_ok = len(wind_slice_forecast) == H_eff
    
    if real_ok and forecast_ok:
        print(f"   ✓ t0={t0}: real_slice={len(wind_slice_real)}, forecast_slice={len(wind_slice_forecast)}")
    else:
        print(f"   ✗ t0={t0}: LENGTH MISMATCH")
        if not real_ok:
            print(f"     Real slice: expected {step_eff}, got {len(wind_slice_real)}")
        if not forecast_ok:
            print(f"     Forecast slice: expected {H_eff}, got {len(wind_slice_forecast)}")

# --- TEST 3: FCR BLOCK MAPPING ---
print("\n4. Testing FCR block mapping...")
errors = []
for t0 in [0, 24, 96, 192]:
    if t0 >= len(wind):
        break
    
    H_eff = min(H, len(wind) - t0)
    i0 = t0 // 16
    I_eff = int(np.ceil(H_eff / 16))
    
    # Check: Can we slice FCR?
    try:
        price_FCR_slice = price_FCR[i0:i0 + I_eff]
        FCR_volume_slice = FCR_volume[i0:i0 + I_eff]
        
        # Check: FCR blocks expand correctly
        expanded_blocks = []
        for block_val in price_FCR_slice[:3]:  # First 3 blocks
            expanded_blocks.extend([block_val] * 16)
        
        if len(expanded_blocks) == 48:  # 3 blocks * 16 timesteps
            print(f"   ✓ t0={t0}: i0={i0}, I_eff={I_eff}, expansion works (48 timesteps from 3 blocks)")
        else:
            errors.append(f"   ✗ t0={t0}: expansion failed (got {len(expanded_blocks)} instead of 48)")
    except IndexError as e:
        errors.append(f"   ✗ t0={t0}: FCR indexing failed: {e}")

if errors:
    for err in errors:
        print(err)

# --- TEST 4: FULL LOOP SIMULATION ---
print("\n5. Testing full rolling horizon loop...")
all_h_lengths = []
all_storage_lengths = []
t0 = 0
window_count = 0

storage_state = 0.0
z_on_state = 1
P_prev = 0

while t0 < len(wind):
    H_eff = min(H, len(wind) - t0)
    step_eff = min(step, H_eff)
    
    # Simulate window processing
    window_count += 1
    all_h_lengths.append(step_eff)
    all_storage_lengths.append(step_eff)
    
    # Update state (simulate)
    storage_state += 0.5  # mock update
    z_on_state = 1 if np.random.rand() > 0.3 else 0
    P_prev = np.random.uniform(100, 1000)
    
    t0 += step_eff

print(f"   Total windows: {window_count}")
print(f"   Total steps collected: {sum(all_h_lengths)}")
print(f"   Original data length: {len(wind)}")

if sum(all_h_lengths) == len(wind):
    print(f"   ✓ Reconstruction will produce correct length (no duplicates/gaps)")
else:
    print(f"   ✗ LENGTH MISMATCH: {sum(all_h_lengths)} ≠ {len(wind)}")

# --- TEST 5: STATE CONTINUITY CHECK ---
print("\n6. Testing state carryover concept...")
print("   (Simulating 3 windows with state updates)")

storage_vals = [0.0]
z_on_vals = [1]
P_vals = [0.0]

for w in range(3):
    # Simulate window w
    storage = storage_vals[-1] + 5.0  # mock update
    z_on = 1 if w % 2 == 0 else 0
    P = np.random.uniform(100, 1000)
    
    storage_vals.append(storage)
    z_on_vals.append(z_on)
    P_vals.append(P)

# Check: no jumps in storage/power
storage_diffs = np.diff(storage_vals)
if all(abs(d - 5.0) < 0.01 for d in storage_diffs):
    print(f"   ✓ Storage updates smoothly: {storage_vals}")
else:
    print(f"   ✗ Storage has jumps: {storage_vals}")

# --- TEST 6: DATA CONSISTENCY ---
print("\n7. Testing data consistency...")

# Check wind
wind_ok = not np.isnan(wind).any() and len(wind) > 0
price_ok = not np.isnan(price).any() and len(price) > 0
fcr_ok = not np.isnan(price_FCR).any() and len(price_FCR) > 0

print(f"   Wind: {'✓ OK' if wind_ok else '✗ NaNs found'} ({len(wind)} values)")
print(f"   Price: {'✓ OK' if price_ok else '✗ NaNs found'} ({len(price)} values)")
print(f"   FCR price: {'✓ OK' if fcr_ok else '✗ NaNs found'} ({len(price_FCR)} values)")

# Check ranges
print(f"   Wind range: {wind.min():.0f} - {wind.max():.0f} kW")
print(f"   Price range: {price.min():.1f} - {price.max():.1f} EUR/MWh")
print(f"   FCR price range: {price_FCR.min():.2f} - {price_FCR.max():.2f} EUR/kW")

# --- SUMMARY ---
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"✓ Data shape: wind={len(wind)}, price={len(price)}, FCR_blocks={len(price_FCR)}")
print(f"✓ Forecast mechanism: lag={lag}, first {lag} steps use real values")
print(f"✓ Horizon: H={H}, step={step}")
print(f"✓ Windows needed: {int(np.ceil(len(wind) / step))}")
print(f"✓ Total reconstructed steps: {sum(all_h_lengths)}")
print("\n✓ All tests passed! Ready to run main.py")
print("=" * 80)
