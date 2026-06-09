from Data_preparation import DataPreparationCSV
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xml.etree.ElementTree as ET
from scipy.interpolate import interp1d
from scipy import stats
import os

# =============================================================
# DATA PREPARATION
# =============================================================
print("Preparing data...")
prep = DataPreparationCSV(
    start_date="2025-03-05",
    end_date="2026-03-05",
    el_price_file_name="DayAheadPrices.csv",
    wind_file_name="WindProd5MinRealtime.csv",
    fcr_file_name="FcrDK1.csv",
    imbalance_file_name="ImbalancePrice.csv"
)
data = prep.data

wind_df       = data['wind']
el_price_df   = data['el_price']
fcr_df        = data['fcr']
imbalance_df  = data['imbalance']
imbalance_price = imbalance_df['ImbalancePriceEUR'].values
print("Data prepared.")

negative_prices_count = (el_price_df['el_price[EUR/MWh]'] < 0).sum()
print(f"Number of timesteps with negative prices: {negative_prices_count}")


# =============================================================
# LOAD ECMWF WIND DATA WITH POWER CURVE + LOSSES
# (independent of DataPreparationCSV for analysis flexibility)
# =============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Loss factors (same as data_preparation.py) ---
WAKE_LOSS         = 0.06
AVAILABILITY_LOSS = 0.05
ELECTRICAL_LOSS   = 0.015
LOSS_FACTOR = (1 - WAKE_LOSS) * (1 - AVAILABILITY_LOSS) * (1 - ELECTRICAL_LOSS)

FARM_CAPACITY_KW = 208_000.0   # kW
TURBINE_RATED_KW = 8_000.0     # kW
N_TURBINES       = FARM_CAPACITY_KW / TURBINE_RATED_KW   # 26

# --- Parse WAsP power curve ---
wtg_path = os.path.join(BASE_DIR, "..", "data", "Vestas_V164-8MW.wtg")
tree = ET.parse(wtg_path)
root = tree.getroot()

rotor_diameter = float(root.get('RotorDiameter'))
description    = root.get('Description')
sss            = root.find('.//StartStopStrategy')
v_cutin        = float(sss.get('LowSpeedCutIn'))
v_cutout       = float(sss.get('HighSpeedCutOut'))
perf           = root.find('.//PerformanceTable')
rho_ref        = float(perf.get('AirDensity'))

records = []
for dp in root.findall('.//DataPoint'):
    ws  = float(dp.get('WindSpeed'))
    pwr = float(dp.get('PowerOutput')) / 1000.0   # W → kW
    records.append((ws, pwr))

df_pc = pd.DataFrame(records, columns=['wind_speed_ms', 'power_kW'])
df_pc = df_pc.sort_values('wind_speed_ms').reset_index(drop=True)
df_pc = pd.concat([
    pd.DataFrame({'wind_speed_ms': [0.0, v_cutin - 0.01], 'power_kW': [0.0, 0.0]}),
    df_pc,
    pd.DataFrame({'wind_speed_ms': [v_cutout + 0.01], 'power_kW': [0.0]})
], ignore_index=True).sort_values('wind_speed_ms').reset_index(drop=True)

pc_interp = interp1d(
    df_pc['wind_speed_ms'], df_pc['power_kW'],
    kind='linear', bounds_error=False, fill_value=0.0
)

print(f"\nTurbine     : {description}")
print(f"Cut-in/out  : {v_cutin} / {v_cutout} m/s")
print(f"Rated power : {df_pc['power_kW'].max():,.0f} kW")
print(f"Loss factor : {LOSS_FACTOR:.4f}  ({(1-LOSS_FACTOR)*100:.1f}% total loss)")

# --- Plot power curve ---
fig, ax = plt.subplots(figsize=(8, 4))
ws_plot = np.linspace(0, 27, 300)
ax.plot(ws_plot, pc_interp(ws_plot) / 1000, lw=2, color='#1976D2')
ax.scatter(df_pc['wind_speed_ms'], df_pc['power_kW'] / 1000,
           s=20, color='#1976D2', zorder=5)
ax.set_xlabel("Wind speed [m/s]")
ax.set_ylabel("Power [MW]")
ax.set_title(f"V164-8MW power curve  (rated: 8 MW, cut-in: {v_cutin} m/s, "
             f"cut-out: {v_cutout} m/s)")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# --- Load ECMWF wind speed CSV ---
ws_path = os.path.join(BASE_DIR, "..", "data", "windspeeds.csv")

with open(ws_path, 'r') as f:
    lines = f.readlines()
header_row = next(i for i, l in enumerate(lines) if l.startswith('time'))

df_ecmwf = pd.read_csv(ws_path, skiprows=header_row, parse_dates=['time'])
df_ecmwf.columns = ['time', 'wind_speed_ms']
df_ecmwf.set_index('time', inplace=True)
df_ecmwf.index = (
    df_ecmwf.index
    .tz_localize('UTC')
    .tz_convert('Europe/Copenhagen')
    .tz_localize(None)
)
df_ecmwf.sort_index(inplace=True)
df_ecmwf = df_ecmwf[~df_ecmwf.index.duplicated(keep='first')]

print(f"\nECMWF records : {len(df_ecmwf)} hourly")
print(f"Date range    : {df_ecmwf.index[0]}  →  {df_ecmwf.index[-1]}")
print(f"Mean wind     : {df_ecmwf['wind_speed_ms'].mean():.2f} m/s")

# --- Hub height correction ---
Z_ECMWF, Z_HUB, Z0 = 100.0, 103.6, 0.0002
correction = np.log(Z_HUB / Z0) / np.log(Z_ECMWF / Z0)
df_ecmwf['wind_speed_hub'] = df_ecmwf['wind_speed_ms'] * correction
print(f"Hub height correction (100m → 103.6m): {correction:.4f}")

# --- Apply power curve + losses ---
df_ecmwf['wind_power_gross_kW'] = (
    pc_interp(df_ecmwf['wind_speed_hub'].values) * N_TURBINES
)
df_ecmwf['wind_power_net_kW'] = (
    df_ecmwf['wind_power_gross_kW'] * LOSS_FACTOR
).clip(lower=0.0)

cf_gross = df_ecmwf['wind_power_gross_kW'].mean() / FARM_CAPACITY_KW * 100
cf_net   = df_ecmwf['wind_power_net_kW'].mean()   / FARM_CAPACITY_KW * 100
print(f"\nECMWF gross capacity factor : {cf_gross:.1f}%")
print(f"ECMWF net  capacity factor  : {cf_net:.1f}%  (after losses)")

# --- Upsample to 15-min ---
df_ecmwf_clean = df_ecmwf[~df_ecmwf.index.duplicated(keep='first')]
df_ecmwf_15min = df_ecmwf_clean[
    ['wind_speed_hub', 'wind_power_gross_kW', 'wind_power_net_kW']
].resample('15min').interpolate(method='linear').clip(lower=0.0)
df_ecmwf_15min['hour'] = df_ecmwf_15min.index.hour

print(f"After upsampling: {len(df_ecmwf_15min)} records at 15-min resolution")


# =============================================================
# PRICE DATA
# =============================================================
da_price    = el_price_df['el_price[EUR/MWh]'].values
imb_price   = imbalance_df['ImbalancePriceEUR'].values
fcr_price   = fcr_df['price_FCR'].values * 1000   # EUR/MW/h
wind_kw     = wind_df['wind_kW'].values

N = min(len(da_price), len(imb_price), len(wind_kw))
da_price  = da_price[:N]
imb_price = imb_price[:N]
wind_kw   = wind_kw[:N]
fcr_price_ts = np.repeat(fcr_price, 16)[:N]

da_index = el_price_df.index[:N]
df_analysis = pd.DataFrame({
    'da_price':  da_price,
    'imb_price': imb_price,
    'wind_kw':   wind_kw,
}, index=da_index)
df_analysis['month'] = df_analysis.index.month
df_analysis['hour']  = df_analysis.index.hour

month_names    = ['Jan','Feb','Mar','Apr','May','Jun',
                  'Jul','Aug','Sep','Oct','Nov','Dec']
months_present = sorted(df_analysis['month'].unique())
dt = 0.25


# =============================================================
# 1. DESCRIPTIVE STATISTICS
# =============================================================
def describe_price(arr, name, unit):
    q1, q99 = np.percentile(arr, 1), np.percentile(arr, 99)
    print(f"\n  {name} [{unit}]")
    print(f"    Mean         : {arr.mean():>10.2f}")
    print(f"    Median       : {np.median(arr):>10.2f}")
    print(f"    Std dev      : {arr.std():>10.2f}")
    print(f"    Min          : {arr.min():>10.2f}")
    print(f"    1st pct      : {q1:>10.2f}")
    print(f"    25th pct     : {np.percentile(arr, 25):>10.2f}")
    print(f"    75th pct     : {np.percentile(arr, 75):>10.2f}")
    print(f"    99th pct     : {q99:>10.2f}")
    print(f"    Max          : {arr.max():>10.2f}")
    print(f"    Negative %   : {100*(arr < 0).mean():>9.1f} %")
    print(f"    > 99th pct % : {100*(arr > q99).mean():>9.1f} %")

print("=" * 50)
print("  DESCRIPTIVE STATISTICS")
print("=" * 50)
describe_price(da_price,  "Day-ahead price",  "EUR/MWh")
describe_price(imb_price, "Imbalance price",  "EUR/MWh")
describe_price(fcr_price, "FCR price",        "EUR/MW/h")

#Winsorization of imbalance price descriptive statistics
imb_price_wins = np.clip(imb_price, np.percentile(imb_price, 1), np.percentile(imb_price, 99))
describe_price(imb_price_wins, "Imbalance price (winsorized)", "EUR/MWh")

#Imbalance price descriptive statistics win the 8 maximum prices winsorized
imb_price_wins_top8 = np.copy(imb_price)
top8_threshold = np.sort(imb_price)[-8]
imb_price_wins_top8[imb_price > top8_threshold] = top8_threshold
describe_price(imb_price_wins_top8, "Imbalance price (top 8 winsorized)", "EUR/MWh")

#find the number of winsorized values
n_wins = ((imb_price < np.percentile(imb_price, 1)) | (imb_price > np.percentile(imb_price, 99))).sum()
print(f"  Number of winsorized values: {n_wins} out of {len(imb_price)} ")

#scatter plot of imbalance prices
plt.figure(figsize=(10, 4))
plt.scatter(range(len(imb_price)), imb_price, s=10, alpha=0.5, color='#E53935')
plt.axhline(imb_price.mean(), color='black', ls='--', lw=1.2, label=f"Mean: {imb_price.mean():.1f} EUR/MWh")
plt.xlabel("Timestep index")
plt.ylabel("Imbalance price [EUR/MWh]")
plt.title("Imbalance price over time")
plt.legend(fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# =============================================================
# 2. PRICE DURATION CURVES
# =============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Price duration curves", fontsize=13, fontweight='bold')

for ax, arr, label, unit, color in zip(
    axes,
    [da_price, imb_price, fcr_price],
    ["Day-ahead", "Imbalance", "FCR"],
    ["EUR/MWh",  "EUR/MWh",   "EUR/MW/h"],
    ["#1976D2",  "#E53935",   "#388E3C"]
):
    sorted_arr = np.sort(arr)[::-1]
    pct        = np.linspace(0, 100, len(sorted_arr))
    ax.plot(pct, sorted_arr, color=color, lw=1.5)
    ax.axhline(0, color='black', lw=0.6, ls='--', alpha=0.5)
    ax.axhline(arr.mean(), color=color, lw=1, ls=':',
               label=f"Mean: {arr.mean():.1f}")
    ax.set_xlabel("% of time exceeded")
    ax.set_ylabel(unit)
    ax.set_title(label)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

#Histogram of DA prices
plt.figure(figsize=(10, 4))
plt.hist(da_price, bins=100, color='#1976D2', alpha=0.7, edgecolor='none')
plt.axvline(da_price.mean(), color='black', ls='--', lw=1.2, label=f"Mean: {da_price.mean():.1f} EUR/MWh")
plt.xlabel("Day-ahead price [EUR/MWh]")
plt.ylabel("Count")
plt.title("Day-ahead price distribution")
plt.legend(fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#Find out when the top 6 imbalance prices occur
top6_idx = np.argsort(imb_price)[-6:]
print("\nTop 6 imbalance price episodes:")
for idx in top6_idx:
    print(f"  {df_analysis.index[idx]} : {imb_price[idx]:.1f} EUR/MWh (DA price: {da_price[idx]:.1f} EUR/MWh)")

#Imbalance prices on 2025-06-02
date_mask = (df_analysis.index >= '2025-06-02') & (df_analysis.index < '2025-06-03')
print("\nImbalance prices on 2025-06-02:")
for idx in np.where(date_mask)[0]:
    print(f"  {df_analysis.index[idx]} : {imb_price[idx]:.1f} EUR/MWh (DA price: {da_price[idx]:.1f} EUR/MWh)")

# =============================================================
# 3. IMBALANCE TAIL ANALYSIS
# =============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Imbalance price tail analysis", fontsize=13, fontweight='bold')

axes[0].hist(imb_price, bins=200, color='#E53935', alpha=0.7, edgecolor='none')
axes[0].axvline(np.percentile(imb_price, 99), color='black', ls='--', lw=1,
                label=f"99th pct: {np.percentile(imb_price,99):.0f}")
axes[0].axvline(np.percentile(imb_price,  1), color='gray',  ls='--', lw=1,
                label=f"1st pct:  {np.percentile(imb_price, 1):.0f}")
axes[0].set_xlabel("Imbalance price [EUR/MWh]")
axes[0].set_ylabel("Count")
axes[0].set_title("Full distribution")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

p1, p99 = np.percentile(imb_price, 1), np.percentile(imb_price, 99)
clipped  = imb_price[(imb_price >= p1) & (imb_price <= p99)]
axes[1].hist(clipped, bins=100, color='#E53935', alpha=0.7, edgecolor='none')
axes[1].axvline(da_price.mean(), color='#1976D2', ls='--', lw=1.2,
                label=f"Mean DA: {da_price.mean():.1f}")
axes[1].axvline(imb_price.mean(), color='black', ls=':', lw=1.2,
                label=f"Mean imb: {imb_price.mean():.1f}")
axes[1].set_xlabel("Imbalance price [EUR/MWh]")
axes[1].set_title("Distribution (1st–99th pct)")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

(osm, osr), (slope, intercept, r) = stats.probplot(imb_price, dist="norm")
axes[2].scatter(osm, osr, s=2, alpha=0.3, color='#E53935')
x_line = np.array([osm.min(), osm.max()])
axes[2].plot(x_line, slope * x_line + intercept, 'k--', lw=1)
axes[2].set_xlabel("Theoretical quantiles (normal)")
axes[2].set_ylabel("Sample quantiles")
axes[2].set_title(f"Q-Q plot vs normal  (R²={r**2:.3f})")
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\n=== IMBALANCE PRICE TAIL SUMMARY ===")
for pct in [90, 95, 99, 99.5, 99.9]:
    v = np.percentile(imb_price, pct)
    print(f"  {pct:5.1f}th pct : {v:>8.1f} EUR/MWh  "
          f"({100*(imb_price > v).mean():.2f}% of time exceeded)")
print(f"  Max       : {imb_price.max():>8.1f} EUR/MWh")
print(f"  Min       : {imb_price.min():>8.1f} EUR/MWh")
print(f"  Negative  : {100*(imb_price < 0).mean():.1f}% of timesteps")


# =============================================================
# 4. DA vs IMBALANCE RELATIONSHIP
# =============================================================
spread = imb_price - da_price

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Day-ahead vs imbalance relationship", fontsize=13, fontweight='bold')

axes[0].scatter(da_price, imb_price, s=2, alpha=0.15, color='#7B1FA2')
lim = max(abs(da_price).max(), abs(imb_price).max()) * 1.05
axes[0].plot([-lim, lim], [-lim, lim], 'k--', lw=0.8, label='1:1 line')
axes[0].set_xlabel("DA price [EUR/MWh]")
axes[0].set_ylabel("Imbalance price [EUR/MWh]")
axes[0].set_title(f"Scatter  (corr={np.corrcoef(da_price, imb_price)[0,1]:.3f})")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

axes[1].hist(spread, bins=150, color='#7B1FA2', alpha=0.7, edgecolor='none')
axes[1].axvline(0, color='black', lw=0.8, ls='--')
axes[1].axvline(spread.mean(), color='orange', lw=1.2,
                label=f"Mean spread: {spread.mean():.1f}")
axes[1].set_xlabel("Imbalance − DA price [EUR/MWh]")
axes[1].set_ylabel("Count")
axes[1].set_title("Price spread distribution")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

da_bins    = np.percentile(da_price, np.linspace(0, 100, 11))
bin_labels = [f"{da_bins[i]:.0f}–{da_bins[i+1]:.0f}" for i in range(10)]
bin_means, bin_stds = [], []
for i in range(10):
    mask   = (da_price >= da_bins[i]) & (da_price < da_bins[i+1])
    subset = spread[mask]
    bin_means.append(subset.mean() if len(subset) > 0 else 0)
    bin_stds.append(subset.std()   if len(subset) > 0 else 0)

x = np.arange(10)
axes[2].bar(x, bin_means, yerr=bin_stds, capsize=3, color='#7B1FA2',
            alpha=0.7, error_kw={'elinewidth': 0.8})
axes[2].axhline(0, color='black', lw=0.7, ls='--')
axes[2].set_xticks(x)
axes[2].set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=8)
axes[2].set_xlabel("DA price decile [EUR/MWh]")
axes[2].set_ylabel("Mean spread ± std [EUR/MWh]")
axes[2].set_title("Spread by DA price level")
axes[2].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print(f"\n  Mean imbalance−DA spread : {spread.mean():>8.2f} EUR/MWh")
print(f"  Std of spread            : {spread.std():>8.2f} EUR/MWh")
print(f"  Imb > DA (%)             : {100*(imb_price > da_price).mean():.1f}%")
print(f"  Imb < DA (%)             : {100*(imb_price < da_price).mean():.1f}%")
print(f"  |Spread| > 50 EUR/MWh    : {100*(np.abs(spread) > 50).mean():.1f}%")


# =============================================================
# 5. FCR PRICE ANALYSIS
# =============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("FCR price analysis", fontsize=13, fontweight='bold')

axes[0].hist(fcr_price, bins=60, color='#388E3C', alpha=0.7, edgecolor='none')
axes[0].axvline(fcr_price.mean(), color='black', ls='--', lw=1.2,
                label=f"Mean: {fcr_price.mean():.1f} EUR/MW/h")
axes[0].set_xlabel("FCR price [EUR/MW/h]")
axes[0].set_ylabel("Count (4h blocks)")
axes[0].set_title("Distribution")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

axes[1].plot(fcr_price, color='#388E3C', lw=0.8, alpha=0.8)
axes[1].axhline(fcr_price.mean(), color='black', ls='--', lw=1,
                label=f"Mean: {fcr_price.mean():.1f}")
if len(fcr_price) >= 180:
    roll = pd.Series(fcr_price).rolling(180, min_periods=1).mean().values
    axes[1].plot(roll, color='orange', lw=1.5, label="30-day rolling mean")
axes[1].set_xlabel("4h block index")
axes[1].set_ylabel("FCR price [EUR/MW/h]")
axes[1].set_title("FCR price over time")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

n_blocks    = len(fcr_price)
block_hours = np.array([(i % 6) * 4 for i in range(n_blocks)])
hour_labels = [0, 4, 8, 12, 16, 20]
fcr_by_hour = [fcr_price[block_hours == h] for h in hour_labels]
axes[2].boxplot(fcr_by_hour,
                tick_labels=[f"{h:02d}:00" for h in hour_labels],
                patch_artist=True,
                boxprops=dict(facecolor='#C8E6C9', color='#388E3C'),
                medianprops=dict(color='#1B5E20', lw=1.5),
                whiskerprops=dict(color='#388E3C'),
                capprops=dict(color='#388E3C'),
                flierprops=dict(marker='.', ms=3, alpha=0.4, color='#388E3C'))
axes[2].set_xlabel("Block start hour")
axes[2].set_ylabel("FCR price [EUR/MW/h]")
axes[2].set_title("FCR price by time of day")
axes[2].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()


# =============================================================
# 6. SEASONALITY — monthly box plots
# =============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Seasonal price patterns (monthly)", fontsize=13, fontweight='bold')

for ax, col, label, color in zip(
    axes,
    ['da_price', 'imb_price'],
    ['DA price [EUR/MWh]', 'Imbalance price [EUR/MWh]'],
    ['#1976D2', '#E53935']
):
    data_by_month = [df_analysis.loc[df_analysis['month'] == m, col].values
                     for m in months_present]
    ax.boxplot(data_by_month,
               tick_labels=[month_names[m-1] for m in months_present],
               patch_artist=True,
               showfliers=False,
               boxprops=dict(facecolor=color+'33', color=color),
               medianprops=dict(color=color, lw=2),
               whiskerprops=dict(color=color),
               capprops=dict(color=color))
    ax.axhline(0, color='black', lw=0.6, ls='--', alpha=0.5)
    ax.set_ylabel(label)
    ax.set_title(label.split('[')[0].strip())
    ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()


# =============================================================
# 7. DIURNAL PATTERNS
# =============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Diurnal price patterns (mean by hour)", fontsize=13, fontweight='bold')

hours = np.arange(24)
for ax, col, label, color in zip(
    axes,
    ['da_price', 'imb_price', 'wind_kw'],
    ['DA price [EUR/MWh]', 'Imbalance price [EUR/MWh]', 'Wind [kW]'],
    ['#1976D2', '#E53935', '#388E3C']
):
    hourly_mean = np.array([df_analysis.loc[df_analysis['hour']==h, col].mean()
                             for h in hours])
    hourly_std  = np.array([df_analysis.loc[df_analysis['hour']==h, col].std()
                             for h in hours])
    ax.plot(hours, hourly_mean, color=color, lw=2)
    ax.fill_between(hours, hourly_mean - hourly_std,
                    hourly_mean + hourly_std, alpha=0.15, color=color)
    ax.axhline(0, color='black', lw=0.5, ls='--', alpha=0.5)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel(label)
    ax.set_title(label.split('[')[0].strip())
    ax.set_xticks([0, 4, 8, 12, 16, 20])
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# =============================================================
# 8. NEGATIVE PRICE ANALYSIS
# =============================================================
neg_da  = df_analysis[df_analysis['da_price']  < 0]
neg_imb = df_analysis[df_analysis['imb_price'] < 0]

print("\n=== NEGATIVE PRICE EPISODES ===")
print(f"  DA price < 0   : {len(neg_da):>5} timesteps  "
      f"({100*len(neg_da)/N:.1f}%)   mean wind: "
      f"{neg_da['wind_kw'].mean():.0f} kW  "
      f"vs overall {wind_kw.mean():.0f} kW")
print(f"  Imb price < 0  : {len(neg_imb):>5} timesteps  "
      f"({100*len(neg_imb)/N:.1f}%)   mean wind: "
      f"{neg_imb['wind_kw'].mean():.0f} kW")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Negative price episodes", fontsize=13, fontweight='bold')

for ax, df_neg, label, color in zip(
    axes,
    [neg_da, neg_imb],
    ['DA price < 0', 'Imbalance price < 0'],
    ['#1976D2', '#E53935']
):
    neg_by_hour = [len(df_neg[df_neg['hour'] == h]) for h in range(24)]
    ax.bar(range(24), neg_by_hour, color=color, alpha=0.75)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Count of negative price timesteps")
    ax.set_title(label)
    ax.set_xticks([0, 4, 8, 12, 16, 20])
    ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()


# =============================================================
# 9. WINSORIZATION SENSITIVITY
# =============================================================
print("\n=== WINSORIZATION SENSITIVITY ===")
print(f"  {'Clip level':>12}  {'Mean':>8}  {'Std':>8}  "
      f"{'Values clipped':>16}  {'Revenue impact':>16}")

for clip_pct in [95, 97, 99, 99.5, 100]:
    if clip_pct == 100:
        clipped   = imb_price.copy()
        n_clipped = 0
    else:
        lo = np.percentile(imb_price, 100 - clip_pct)
        hi = np.percentile(imb_price, clip_pct)
        clipped   = np.clip(imb_price, lo, hi)
        n_clipped = int(np.sum((imb_price < lo) | (imb_price > hi)))
    diff_eur = np.sum((imb_price - clipped) * dt / 1000)
    print(f"  {str(clip_pct)+'th pct':>12}  {clipped.mean():>8.1f}  "
          f"{clipped.std():>8.1f}  {n_clipped:>16}  "
          f"{diff_eur:>+14.0f} EUR/MW")


# =============================================================
# 10. ECMWF WIND — GROSS VS NET DURATION CURVES
# =============================================================
sorted_gross = np.sort(df_ecmwf_15min['wind_power_gross_kW'].values)[::-1] / 1000
sorted_net   = np.sort(df_ecmwf_15min['wind_power_net_kW'].values)[::-1]   / 1000
pct          = np.linspace(0, 100, len(sorted_gross))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("ECMWF IFS wind power duration curves — gross vs net",
             fontweight='bold')

axes[0].plot(pct, sorted_gross, color='#1976D2', lw=1.5,
             label=f'Gross  (CF={cf_gross:.1f}%)')
axes[0].plot(pct, sorted_net,   color='#388E3C', lw=1.5,
             label=f'Net  (CF={cf_net:.1f}%,  '
                   f'{(1-LOSS_FACTOR)*100:.0f}% losses)')
axes[0].axhline(FARM_CAPACITY_KW / 1e6 * 1000, color='gray', ls=':', lw=1,
                label=f'Rated {FARM_CAPACITY_KW/1000:.0f} MW')
axes[0].set_xlabel("% of time exceeded")
axes[0].set_ylabel("Wind power [MW]")
axes[0].set_title("Full duration curve")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

axes[1].plot(pct, sorted_gross, color='#1976D2', lw=1.5, label='Gross')
axes[1].plot(pct, sorted_net,   color='#388E3C', lw=1.5, label='Net')
axes[1].axhline(FARM_CAPACITY_KW / 1000, color='gray', ls=':', lw=1)
axes[1].set_xlim(0, 35)
axes[1].set_xlabel("% of time exceeded")
axes[1].set_ylabel("Wind power [MW]")
axes[1].set_title("Zoomed: 0–35% (rated power region)")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()


# =============================================================
# 11. ECMWF vs SCADA COMPARISON
# =============================================================
common_idx = df_ecmwf_15min.index.intersection(wind_df.index)
ecmwf_arr  = df_ecmwf_15min.loc[common_idx, 'wind_power_net_kW'].values
scada_arr  = wind_df.loc[common_idx, 'wind_kW'].values

corr = np.corrcoef(ecmwf_arr, scada_arr)[0, 1]
mae  = np.abs(ecmwf_arr - scada_arr).mean()
rmse = np.sqrt(((ecmwf_arr - scada_arr) ** 2).mean())
bias = (ecmwf_arr - scada_arr).mean()
cf_scada = scada_arr.mean() / FARM_CAPACITY_KW * 100

print("\n=== ECMWF (net) vs SCADA COMPARISON ===")
print(f"  Correlation            : {corr:.3f}")
print(f"  MAE                    : {mae:,.0f} kW  ({mae/FARM_CAPACITY_KW*100:.1f}% of rated)")
print(f"  RMSE                   : {rmse:,.0f} kW  ({rmse/FARM_CAPACITY_KW*100:.1f}% of rated)")
print(f"  Mean bias (ECMWF−SCADA): {bias:+,.0f} kW")
print(f"  SCADA capacity factor  : {cf_scada:.1f}%")
print(f"  ECMWF net CF           : {cf_net:.1f}%")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("ECMWF IFS (net) vs SCADA — V164-8MW power curve",
             fontsize=13, fontweight='bold')

n_week = 96 * 7
axes[0, 0].plot(ecmwf_arr[:n_week] / 1000, lw=1, color='#E53935',
                label='ECMWF net', alpha=0.9)
axes[0, 0].plot(scada_arr[:n_week] / 1000, lw=1, color='#1976D2',
                label='SCADA', alpha=0.9)
axes[0, 0].set_ylabel("Power [MW]")
axes[0, 0].set_title("First week: ECMWF net vs SCADA")
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

lim = FARM_CAPACITY_KW / 1000 * 1.05
axes[0, 1].scatter(scada_arr / 1000, ecmwf_arr / 1000,
                   s=1, alpha=0.08, color='#388E3C')
axes[0, 1].plot([0, lim], [0, lim], 'k--', lw=0.8, label='1:1')
axes[0, 1].set_xlabel("SCADA power [MW]")
axes[0, 1].set_ylabel("ECMWF net power [MW]")
axes[0, 1].set_title(f"Scatter  (corr={corr:.3f}, bias={bias/1000:+.1f} MW)")
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(alpha=0.3)

diff = ecmwf_arr - scada_arr
axes[1, 0].hist(diff / 1000, bins=100, color='#7B1FA2', alpha=0.7, edgecolor='none')
axes[1, 0].axvline(0,          color='black',   lw=0.8, ls='--')
axes[1, 0].axvline(bias / 1000, color='#7B1FA2', lw=1.5,
                   label=f'Mean bias: {bias/1000:+.1f} MW')
axes[1, 0].set_xlabel("ECMWF net − SCADA [MW]")
axes[1, 0].set_ylabel("Count")
axes[1, 0].set_title("Difference distribution")
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

df_comp = pd.DataFrame({'ecmwf_kW': ecmwf_arr, 'scada_kW': scada_arr},
                        index=common_idx)
df_comp['month'] = df_comp.index.month
months_c   = sorted(df_comp['month'].unique())
cf_ecmwf_m = [df_comp.loc[df_comp['month']==m, 'ecmwf_kW'].mean()
               / FARM_CAPACITY_KW * 100 for m in months_c]
cf_scada_m = [df_comp.loc[df_comp['month']==m, 'scada_kW'].mean()
               / FARM_CAPACITY_KW * 100 for m in months_c]

x = np.arange(len(months_c))
w = 0.35
axes[1, 1].bar(x - w/2, cf_ecmwf_m, width=w, label='ECMWF net',
               color='#E53935', alpha=0.8)
axes[1, 1].bar(x + w/2, cf_scada_m, width=w, label='SCADA',
               color='#1976D2', alpha=0.8)
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels([month_names[m-1] for m in months_c])
axes[1, 1].set_ylabel("Capacity factor [%]")
axes[1, 1].set_title("Monthly CF: ECMWF net vs SCADA")
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()


# =============================================================
# 12. ECMWF WIND vs DA PRICE CORRELATION
# =============================================================
common_idx_price = el_price_df.index.intersection(df_ecmwf_15min.index)
da_aligned       = el_price_df.loc[common_idx_price, 'el_price[EUR/MWh]'].values
ecmwf_net_aligned = (df_ecmwf_15min.loc[common_idx_price, 'wind_power_net_kW'].values
                     / 1000)   # MW

corr_da_ecmwf = np.corrcoef(da_aligned, ecmwf_net_aligned)[0, 1]
print(f"\n=== ECMWF NET WIND vs DA PRICE ===")
print(f"  Correlation: {corr_da_ecmwf:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("ECMWF IFS (net) wind power vs day-ahead price",
             fontweight='bold')

axes[0].scatter(ecmwf_net_aligned, da_aligned,
                s=4, alpha=0.3, color='#7B1FA2')
axes[0].set_xlabel("ECMWF net wind power [MW]")
axes[0].set_ylabel("Day-ahead price [EUR/MWh]")
axes[0].set_title(f"Scatter  (corr={corr_da_ecmwf:.3f})")
axes[0].grid(alpha=0.3)

h, xedges, yedges = np.histogram2d(
    ecmwf_net_aligned, da_aligned, bins=[30, 30],
    range=[[0, FARM_CAPACITY_KW / 1000 * 1.05],
           [np.percentile(da_aligned, 1), np.percentile(da_aligned, 99)]]
)
im = axes[1].imshow(h.T, origin='lower', aspect='auto',
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                    cmap='Purples')
plt.colorbar(im, ax=axes[1], label='Count')
axes[1].set_xlabel("ECMWF net wind power [MW]")
axes[1].set_ylabel("Day-ahead price [EUR/MWh]")
axes[1].set_title("Joint distribution")
axes[1].grid(alpha=0.2)

plt.tight_layout()
plt.show()


# =============================================================
# 13. ECMWF DIURNAL PATTERN
# =============================================================
hourly_mean_ecmwf = (df_ecmwf_15min.groupby('hour')['wind_power_net_kW'].mean()
                     / 1000)
hourly_std_ecmwf  = (df_ecmwf_15min.groupby('hour')['wind_power_net_kW'].std()
                     / 1000)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(hourly_mean_ecmwf.index, hourly_mean_ecmwf.values,
        color='#388E3C', lw=2)
ax.fill_between(hourly_mean_ecmwf.index,
                hourly_mean_ecmwf.values - hourly_std_ecmwf.values,
                hourly_mean_ecmwf.values + hourly_std_ecmwf.values,
                alpha=0.15, color='#388E3C')
ax.set_xlabel("Hour of day")
ax.set_ylabel("Mean wind power [MW]")
ax.set_title("Diurnal pattern of ECMWF IFS net wind power")
ax.set_xticks([0, 4, 8, 12, 16, 20])
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# =============================================================
# 14. ECMWF WIND SPEED vs POWER — SCATTER WITH POWER CURVE
# =============================================================
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df_ecmwf_15min['wind_speed_hub'],
           df_ecmwf_15min['wind_power_net_kW'] / 1000,
           s=4, alpha=0.3, color='#388E3C',
           label='ECMWF net power')

# Overlay the gross power curve for reference
ws_range = np.linspace(0, 30, 300)
ax.plot(ws_range, pc_interp(ws_range) * N_TURBINES / 1000,
        color='#E53935', lw=1.5, ls='--',
        label='Gross power curve')
ax.plot(ws_range, pc_interp(ws_range) * N_TURBINES * LOSS_FACTOR / 1000,
        color='#1976D2', lw=1.5, ls='-',
        label=f'Net power curve (×{LOSS_FACTOR:.3f})')

ax.set_xlabel("Hub-height wind speed [m/s]")
ax.set_ylabel("Wind power [MW]")
ax.set_title("ECMWF IFS: wind speed vs net wind power")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# =============================================================
# 15. PERIOD PLOT HELPER
# =============================================================
def plot_ecmwf_period(start, end, col='wind_power_net_kW'):
    s   = pd.Timestamp(start)
    e   = pd.Timestamp(end)
    seg = df_ecmwf_15min.loc[s:e, col] / 1000.0   # MW

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(seg.index, seg.values, color='#388E3C', lw=1,
            label='ECMWF IFS net power')
    ax.set_xlabel("Time")
    ax.set_ylabel("Power [MW]")
    ax.set_title(f"ECMWF IFS net wind power  {s.date()} → {e.date()}")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

plot_ecmwf_period('2025-04-01', '2025-04-30')


def plot_ecmwf_wind_period(start, end, df=df_ecmwf_15min):
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    seg = df.loc[s:e]
    if seg.empty:
        print(f"No data in range {s} → {e}")
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(seg.index, seg['wind_speed_hub'],
            color='#1976D2', lw=1.5, label='Hub-height wind speed [m/s]')
    ax.set_xlabel("Time")
    ax.set_ylabel("Wind speed [m/s]", color='#1976D2')
    ax.tick_params(axis='y', labelcolor='#1976D2')

    ax2 = ax.twinx()
    ax2.plot(seg.index, seg['wind_power_net_kW'] / 1000,
             color='#388E3C', lw=1.5, label='ECMWF IFS net power [MW]')
    ax2.set_ylabel("Net wind power [MW]", color='#388E3C')
    ax2.tick_params(axis='y', labelcolor='#388E3C')

    ax.set_title(f"ECMWF IFS: hub-height wind speed and net wind power  "
                 f"{s.date()} → {e.date()}")
    ax.grid(alpha=0.3)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              loc='upper right', bbox_to_anchor=(0.99, 0.99))

    plt.tight_layout()
    plt.show()

# Usage examples:
plot_ecmwf_wind_period('2025-04-01', '2025-04-30')
plot_ecmwf_wind_period('2025-06-01', '2025-06-14')
plot_ecmwf_wind_period('2025-11-01', '2025-11-30')


#PRINT RAMP RATE CHARACTERISTICS OF ECMWF NET WIND POWER: STD OF 15-MINUTE CHANGES, RAMPS EXCEEDING 5% OF RATED POWER, ETC.
ecmwf_net = df_ecmwf_15min['wind_power_net_kW'].values / 1000  # MW
ecmwf_ramp = np.diff(ecmwf_net)
ramp_std = ecmwf_ramp.std()
ramp_5pct = (np.abs(ecmwf_ramp) > FARM_CAPACITY_KW * 0.05 / 1000).mean() * 100
print(f"\n=== ECMWF NET WIND POWER RAMP CHARACTERISTICS ===")
print(f"  Std of 15-min ramp: {ramp_std:.2f} MW")
print(f"  Ramps > 5% of rated power: {ramp_5pct:.1f}% of 15-min intervals")


#print average hub-height wind speed (df_ecmwf_15min['wind_speed_hub']) at negative DA prices
neg_da_speed = df_ecmwf_15min.loc[df_analysis['da_price'] < 0, 'wind_speed_hub']
print(f"\nAverage hub-height wind speed during negative DA price episodes: "
        f"{neg_da_speed.mean():.2f} m/s")


#plot diurnal FCR price
fcr_price_by_hour = [fcr_price[block_hours == h] for h in hour_labels]
fig, ax = plt.subplots(figsize=(8, 5))
ax.boxplot(fcr_price_by_hour,
           tick_labels=[f"{h:02d}:00" for h in hour_labels],
           patch_artist=True,
           boxprops=dict(facecolor='#C8E6C9', color='#388E3C'),
           medianprops=dict(color='#1B5E20', lw=1.5),
           whiskerprops=dict(color='#388E3C'),
           capprops=dict(color='#388E3C'),
           flierprops=dict(marker='.', ms=3, alpha=0.4, color='#388E3C'))
ax.set_xlabel("Block start hour")
ax.set_ylabel("FCR price [EUR/MW/h]")
ax.set_title("FCR price by time of day")
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.show()