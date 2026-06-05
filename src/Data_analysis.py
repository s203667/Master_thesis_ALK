from Data_preparation import DataPreparationCSV
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

#print mean wind
print("Mean wind production:", np.mean(wind_df['wind_kW'].values))
#print variance of wind
print("Variance of wind production:", np.var(wind_df['wind_kW'].values))

#plot power duration curve
plt.figure(figsize=(10, 6))
plt.plot(np.sort(wind_df['wind_kW'])[::-1], label='Wind Power Production')
plt.xlabel('Timesteps (sorted)')
plt.ylabel('Wind Power (kW)')
plt.title('Wind Power Production Duration Curve')
plt.grid()
plt.legend()
plt.show()




#plot march 2025 wind production
plt.figure(figsize=(10, 6))
plt.plot(wind_df['wind_kW'][:96*31], label='Wind Power Production (mar 2025)')
plt.xlabel('Time')
plt.ylabel('Wind Power (kW)')
plt.title('Wind Power Production in March 2025')
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(wind_df['wind_kW'][:96*31].rolling(96).std(), label='Rolling 24h Std Dev')
plt.title('Rolling 24h Std Dev of Wind Power Production')
plt.xlabel('Time')
plt.ylabel('Std Dev (kW)')
plt.grid()
plt.show()

#Day ahead price analysis
negative_prices_count = (el_price_df['el_price[EUR/MWh]'] < 0).sum()
print(f"Number of timesteps with negative prices: {negative_prices_count}")

#plot histogram of day ahead prices
plt.figure(figsize=(10, 6))
plt.hist(el_price_df['el_price[EUR/MWh]'], bins=100, alpha=0.7)
plt.xlabel('Day Ahead Price (EUR/MWh)')
plt.ylabel('Frequency')
plt.title('Histogram of Day Ahead Prices')
plt.grid()
plt.show()

#Find top 5% of day ahead prices
top_5_percent_threshold = np.percentile(el_price_df['el_price[EUR/MWh]'], 95)
print(f"95th percentile of day ahead prices: {top_5_percent_threshold} EUR/MWh")

#Print corrrelation between DA_price and windpower
windMW = wind_df['wind_kW']/1000
correlation = np.corrcoef(el_price_df['el_price[EUR/MWh]'], windMW[:len(el_price_df)])[0, 1]
print(f"Correlation between DA price and wind power: {correlation}")

#boxplot of day ahead prices
plt.figure(figsize=(10, 6))
plt.boxplot(el_price_df['el_price[EUR/MWh]'], vert=False)
plt.xlabel('Day Ahead Price (EUR/MWh)')
plt.title('Boxplot of Day Ahead Prices')
plt.grid()
plt.show()

#plot a price duration curve for the day ahead prices
plt.figure(figsize=(10, 6))
plt.plot(np.sort(el_price_df['el_price[EUR/MWh]'])[::-1], label='Day Ahead Prices')
plt.xlabel('Time (sorted)')
plt.ylabel('Day Ahead Price (EUR/MWh)')
plt.title('Day Ahead Price Duration Curve')
plt.grid()
plt.legend()
plt.show()


#Plot Day Ahead Price vs Wind Power Production
plt.figure(figsize=(10, 6))
plt.scatter(wind_df['wind_kW'][:35037], el_price_df['el_price[EUR/MWh]'], alpha=0.5)
plt.xlabel('Wind Power Production (kW)')
plt.ylabel('Day Ahead Price (EUR/MWh)')
plt.title('Day Ahead Price vs Wind Power Production')
plt.grid()
plt.show()


#Plot joint distribution of day ahead price and wind power production
plt.figure(figsize=(10, 6))
plt.hist2d(wind_df['wind_kW'][:35037], el_price_df['el_price[EUR/MWh]'], bins=50, cmap='Blues')
plt.colorbar(label='Count in bin')
plt.xlabel('Wind Power Production (kW)')
plt.ylabel('Day Ahead Price (EUR/MWh)')
plt.title('Joint Distribution of Day Ahead Price and Wind Power Production')
plt.grid()
plt.show()


#plot wind spikes and price spices 
spikes = el_price_df[el_price_df['el_price[EUR/MWh]'] > el_price_df['el_price[EUR/MWh]'].quantile(0.99)]
spike_wind = wind_df.loc[spikes.index]
plt.figure(figsize=(10, 6))
plt.scatter(spike_wind['wind_kW'], spikes['el_price[EUR/MWh]'], color='red', label='Price Spikes')
plt.xlabel('Wind Power Production (kW)')
plt.ylabel('Day Ahead Price (EUR/MWh)')
plt.title('Price Spikes vs Wind Power Production')
plt.grid()
plt.legend()
plt.show()

#imbalance prices histogram
plt.figure(figsize=(10, 6))
plt.hist(imbalance_price, bins=100, alpha=0.7)
plt.xlabel('Imbalance Price (EUR/MWh)')
plt.ylabel('Frequency')
plt.title('Histogram of Imbalance Prices')
plt.grid()
plt.show()


#Plot FCR prices histogram
plt.figure(figsize=(10, 6))
plt.hist(fcr_df['price_FCR']*1000, bins=100, alpha=0.7)
plt.xlabel('FCR Price (EUR/MW/h)')
plt.ylabel('Frequency')
plt.title('Histogram of FCR Prices')
plt.grid()
plt.show()

# =============================================================
# PRICE ANALYSIS SESSION
# =============================================================
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import stats

# --- Prepare arrays ---
da_price    = el_price_df['el_price[EUR/MWh]'].values
imb_price   = imbalance_df['ImbalancePriceEUR'].values
fcr_price   = fcr_df['price_FCR'].values * 1000          # EUR/MW/h
wind_kw     = wind_df['wind_kW'].values

# Align lengths (imbalance may differ from DA due to resolution)
N = min(len(da_price), len(imb_price), len(wind_kw))
da_price  = da_price[:N]
imb_price = imb_price[:N]
wind_kw   = wind_kw[:N]

# FCR is 4h blocks — expand to 15-min for alignment
fcr_price_ts = np.repeat(fcr_price, 16)[:N]   # 1 block = 16 × 15min

dt = 0.25   # h


# =============================================================
# 1. DESCRIPTIVE STATISTICS TABLE
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
describe_price(da_price,  "Day-ahead price",   "EUR/MWh")
describe_price(imb_price, "Imbalance price",   "EUR/MWh")
describe_price(fcr_price, "FCR price",         "EUR/MW/h")


# =============================================================
# 2. PRICE DURATION CURVES — all three on one figure
# =============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Price duration curves", fontsize=13, fontweight='bold')

for ax, arr, label, unit, color in zip(
    axes,
    [da_price, imb_price, fcr_price],
    ["Day-ahead", "Imbalance", "FCR"],
    ["EUR/MWh", "EUR/MWh", "EUR/MW/h"],
    ["#1976D2", "#E53935", "#388E3C"]
):
    sorted_arr = np.sort(arr)[::-1]
    pct        = np.linspace(0, 100, len(sorted_arr))
    ax.plot(pct, sorted_arr, color=color, lw=1.5)
    ax.axhline(0, color='black', lw=0.6, ls='--', alpha=0.5)
    ax.axhline(arr.mean(), color=color, lw=1, ls=':', alpha=0.7,
               label=f"Mean: {arr.mean():.1f}")
    ax.set_xlabel("% of time exceeded")
    ax.set_ylabel(unit)
    ax.set_title(label)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# =============================================================
# 3. IMBALANCE PRICE — TAIL ANALYSIS
# =============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Imbalance price tail analysis", fontsize=13, fontweight='bold')

# 3a. Full histogram
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

# 3b. Clipped at 99th/1st — see the body
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

# 3c. QQ plot vs normal
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

# Tail summary
print("\n=== IMBALANCE PRICE TAIL SUMMARY ===")
for pct in [90, 95, 99, 99.5, 99.9]:
    v = np.percentile(imb_price, pct)
    print(f"  {pct:5.1f}th pct : {v:>8.1f} EUR/MWh  "
          f"({100*(imb_price > v).mean():.2f}% of time exceeded)")
print(f"  Max          : {imb_price.max():>8.1f} EUR/MWh")
print(f"  Min          : {imb_price.min():>8.1f} EUR/MWh")
print(f"  Negative     : {100*(imb_price < 0).mean():.1f}% of timesteps")
print(f"  |Imb| > DA mean  : {100*(np.abs(imb_price) > da_price.mean()).mean():.1f}% of timesteps")


# =============================================================
# 4. DA vs IMBALANCE — SPREAD AND RELATIONSHIP
# =============================================================
spread = imb_price - da_price

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Day-ahead vs imbalance relationship", fontsize=13, fontweight='bold')

# 4a. Scatter
axes[0].scatter(da_price, imb_price, s=2, alpha=0.15, color='#7B1FA2')
lim = max(abs(da_price).max(), abs(imb_price).max()) * 1.05
axes[0].plot([-lim, lim], [-lim, lim], 'k--', lw=0.8, label='1:1 line')
axes[0].set_xlabel("DA price [EUR/MWh]")
axes[0].set_ylabel("Imbalance price [EUR/MWh]")
axes[0].set_title(f"Scatter  (corr={np.corrcoef(da_price, imb_price)[0,1]:.3f})")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

# 4b. Spread distribution
axes[1].hist(spread, bins=150, color='#7B1FA2', alpha=0.7, edgecolor='none')
axes[1].axvline(0, color='black', lw=0.8, ls='--')
axes[1].axvline(spread.mean(), color='orange', lw=1.2, ls='-',
                label=f"Mean spread: {spread.mean():.1f}")
axes[1].set_xlabel("Imbalance − DA price [EUR/MWh]")
axes[1].set_ylabel("Count")
axes[1].set_title("Price spread distribution")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

# 4c. Imbalance sign vs DA price level
# Does the imbalance price diverge more when DA is high/low?
da_bins    = np.percentile(da_price, np.linspace(0, 100, 11))
bin_labels = [f"{da_bins[i]:.0f}–{da_bins[i+1]:.0f}" for i in range(10)]
bin_means  = []
bin_stds   = []
for i in range(10):
    mask = (da_price >= da_bins[i]) & (da_price < da_bins[i+1])
    subset = spread[mask]
    bin_means.append(subset.mean() if len(subset) > 0 else 0)
    bin_stds.append(subset.std()   if len(subset) > 0 else 0)

x = np.arange(10)
axes[2].bar(x, bin_means, yerr=bin_stds, capsize=3,
            color='#7B1FA2', alpha=0.7, error_kw={'elinewidth': 0.8})
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

# 5a. Histogram
axes[0].hist(fcr_price, bins=60, color='#388E3C', alpha=0.7, edgecolor='none')
axes[0].axvline(fcr_price.mean(), color='black', ls='--', lw=1.2,
                label=f"Mean: {fcr_price.mean():.1f} EUR/MW/h")
axes[0].set_xlabel("FCR price [EUR/MW/h]")
axes[0].set_ylabel("Count (4h blocks)")
axes[0].set_title("Distribution")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

# 5b. FCR price over time
axes[1].plot(fcr_price, color='#388E3C', lw=0.8, alpha=0.8)
axes[1].axhline(fcr_price.mean(), color='black', ls='--', lw=1,
                label=f"Mean: {fcr_price.mean():.1f}")
# Rolling 30-day mean (30 days * 6 blocks/day = 180 blocks)
if len(fcr_price) >= 180:
    roll = pd.Series(fcr_price).rolling(180, min_periods=1).mean().values
    axes[1].plot(roll, color='orange', lw=1.5, label="30-day rolling mean")
axes[1].set_xlabel("4h block index")
axes[1].set_ylabel("FCR price [EUR/MW/h]")
axes[1].set_title("FCR price over time")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

# 5c. FCR price by hour-of-day (block start hour: 0, 4, 8, 12, 16, 20)
n_blocks      = len(fcr_price)
block_hours   = np.array([(i % 6) * 4 for i in range(n_blocks)])
hour_labels   = [0, 4, 8, 12, 16, 20]
fcr_by_hour   = [fcr_price[block_hours == h] for h in hour_labels]
axes[2].boxplot(fcr_by_hour, labels=[f"{h:02d}:00" for h in hour_labels],
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
# 6. WIND vs PRICES — key for optimization insights
# =============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Wind production vs prices", fontsize=13, fontweight='bold')

wind_pct = wind_kw / wind_kw.max() * 100   # normalised capacity factor %

# 6a. Wind vs DA price — 2D histogram
h, xedges, yedges = np.histogram2d(wind_pct, da_price,
                                    bins=[30, 30],
                                    range=[[0, 100],
                                           [np.percentile(da_price, 1),
                                            np.percentile(da_price, 99)]])
im = axes[0].imshow(h.T, origin='lower', aspect='auto',
                     extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                     cmap='Blues')
plt.colorbar(im, ax=axes[0], label='Count')
axes[0].set_xlabel("Wind capacity factor [%]")
axes[0].set_ylabel("DA price [EUR/MWh]")
axes[0].set_title(f"Wind vs DA  (corr={np.corrcoef(wind_pct, da_price)[0,1]:.3f})")
axes[0].grid(alpha=0.2)

# 6b. Wind vs imbalance price
h2, xedges2, yedges2 = np.histogram2d(wind_pct, imb_price,
                                        bins=[30, 30],
                                        range=[[0, 100],
                                               [np.percentile(imb_price, 1),
                                                np.percentile(imb_price, 99)]])
im2 = axes[1].imshow(h2.T, origin='lower', aspect='auto',
                      extent=[xedges2[0], xedges2[-1], yedges2[0], yedges2[-1]],
                      cmap='Reds')
plt.colorbar(im2, ax=axes[1], label='Count')
axes[1].set_xlabel("Wind capacity factor [%]")
axes[1].set_ylabel("Imbalance price [EUR/MWh]")
axes[1].set_title(f"Wind vs imbalance  (corr={np.corrcoef(wind_pct, imb_price)[0,1]:.3f})")
axes[1].grid(alpha=0.2)

# 6c. Mean DA and imbalance price by wind decile
wind_deciles = np.percentile(wind_pct, np.linspace(0, 100, 11))
dec_centers  = [(wind_deciles[i] + wind_deciles[i+1]) / 2 for i in range(10)]
da_by_decile  = []
imb_by_decile = []
for i in range(10):
    mask = (wind_pct >= wind_deciles[i]) & (wind_pct < wind_deciles[i+1])
    da_by_decile.append(da_price[mask].mean())
    imb_by_decile.append(imb_price[mask].mean())

axes[2].plot(dec_centers, da_by_decile,  'o-', color='#1976D2', lw=1.5,
             label='Mean DA price')
axes[2].plot(dec_centers, imb_by_decile, 's--', color='#E53935', lw=1.5,
             label='Mean imbalance price')
axes[2].axhline(0, color='black', lw=0.6, ls='--', alpha=0.5)
axes[2].set_xlabel("Wind capacity factor [%]")
axes[2].set_ylabel("Price [EUR/MWh]")
axes[2].set_title("Mean prices by wind output level")
axes[2].legend(fontsize=9)
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\n=== WIND-PRICE CORRELATIONS ===")
print(f"  Corr(wind, DA price)       : {np.corrcoef(wind_pct, da_price)[0,1]:>7.3f}")
print(f"  Corr(wind, imb price)      : {np.corrcoef(wind_pct, imb_price)[0,1]:>7.3f}")
print(f"  Corr(wind, FCR price)      : {np.corrcoef(wind_kw[:len(fcr_price_ts)],
                                                      fcr_price_ts)[0,1]:>7.3f}")
print(f"  Corr(DA price, imb price)  : {np.corrcoef(da_price, imb_price)[0,1]:>7.3f}")


# =============================================================
# 7. SEASONALITY — monthly box plots
# =============================================================
# Build a datetime index aligned to DA price
da_index = el_price_df.index[:N]

df_analysis = pd.DataFrame({
    'da_price':  da_price,
    'imb_price': imb_price,
    'wind_kw':   wind_kw,
}, index=da_index)
df_analysis['month'] = df_analysis.index.month
df_analysis['hour']  = df_analysis.index.hour

month_names = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']
months_present = sorted(df_analysis['month'].unique())

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
    bp = ax.boxplot(data_by_month,
                    labels=[month_names[m-1] for m in months_present],
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
# 8. DIURNAL PATTERNS — mean by hour of day
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
    hourly_mean = [df_analysis.loc[df_analysis['hour'] == h, col].mean()
                   for h in hours]
    hourly_std  = [df_analysis.loc[df_analysis['hour'] == h, col].std()
                   for h in hours]
    hourly_mean = np.array(hourly_mean)
    hourly_std  = np.array(hourly_std)

    ax.plot(hours, hourly_mean, color=color, lw=2)
    ax.fill_between(hours,
                    hourly_mean - hourly_std,
                    hourly_mean + hourly_std,
                    alpha=0.15, color=color)
    ax.axhline(0, color='black', lw=0.5, ls='--', alpha=0.5)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel(label)
    ax.set_title(label.split('[')[0].strip())
    ax.set_xticks([0, 4, 8, 12, 16, 20])
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# =============================================================
# 9. NEGATIVE PRICE ANALYSIS — when and why
# =============================================================
neg_da  = df_analysis[df_analysis['da_price']  < 0]
neg_imb = df_analysis[df_analysis['imb_price'] < 0]

print("\n=== NEGATIVE PRICE EPISODES ===")
print(f"  DA price < 0   : {len(neg_da):>5} timesteps  "
      f"({100*len(neg_da)/N:.1f}%)   mean wind during: "
      f"{neg_da['wind_kw'].mean():.0f} kW  "
      f"vs overall mean {wind_kw.mean():.0f} kW")
print(f"  Imb price < 0  : {len(neg_imb):>5} timesteps  "
      f"({100*len(neg_imb)/N:.1f}%)   mean wind during: "
      f"{neg_imb['wind_kw'].mean():.0f} kW")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Negative price episodes", fontsize=13, fontweight='bold')

for ax, df_neg, all_arr, label, color in zip(
    axes,
    [neg_da, neg_imb],
    [da_price, imb_price],
    ['DA price < 0', 'Imbalance price < 0'],
    ['#1976D2', '#E53935']
):
    # Hour-of-day distribution of negative prices
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
# 10. IMBALANCE PRICE WINSORIZATION IMPACT
# =============================================================
# Show what different clipping levels do to the distribution
# — relevant for deciding your optimization assumption

print("\n=== WINSORIZATION SENSITIVITY ===")
print(f"  {'Clip level':>12}  {'Mean':>8}  {'Std':>8}  "
      f"{'Values clipped':>16}  {'Revenue impact est.':>20}")

for clip_pct in [95, 97, 99, 99.5, 100]:
    if clip_pct == 100:
        clipped = imb_price.copy()
        n_clipped = 0
    else:
        lo = np.percentile(imb_price, 100 - clip_pct)
        hi = np.percentile(imb_price, clip_pct)
        clipped   = np.clip(imb_price, lo, hi)
        n_clipped = int(np.sum((imb_price < lo) | (imb_price > hi)))

    # Rough revenue impact: sum of clipped differences * dt * 1 MW hypothetical
    diff_eur = np.sum((imb_price - clipped) * dt / 1000)   # EUR per MW of imbalance
    print(f"  {str(clip_pct)+'th pct':>12}  {clipped.mean():>8.1f}  "
          f"{clipped.std():>8.1f}  {n_clipped:>16}  "
          f"{diff_eur:>+18.0f} EUR/MW")



###########################################################
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

# =============================================================
# 1. PARSE POWER CURVE FROM WAsP FILE
# =============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
# or just BASE_DIR = "." if running in a notebook

wtg_path = os.path.join(BASE_DIR, "..", "data", "Vestas_V164-8MW.wtg")  

tree = ET.parse(wtg_path)
root = tree.getroot()

# Read turbine metadata
rotor_diameter = float(root.get('RotorDiameter'))
description    = root.get('Description')

# Read cut-in / cut-out from StartStopStrategy
sss      = root.find('.//StartStopStrategy')
v_cutin  = float(sss.get('LowSpeedCutIn'))    # 4 m/s
v_cutout = float(sss.get('HighSpeedCutOut'))   # 25 m/s

# Read air density the curve was measured at
perf     = root.find('.//PerformanceTable')
rho_ref  = float(perf.get('AirDensity'))       # 1.225 kg/m³

# Read DataPoints — PowerOutput is in WATTS
records = []
for dp in root.findall('.//DataPoint'):
    ws  = float(dp.get('WindSpeed'))
    pwr = float(dp.get('PowerOutput')) / 1000.0  # W → kW
    records.append((ws, pwr))

df_pc = pd.DataFrame(records, columns=['wind_speed_ms', 'power_kW'])
df_pc = df_pc.sort_values('wind_speed_ms').reset_index(drop=True)

# Add explicit zero at cut-in boundary and just beyond cut-out
df_pc = pd.concat([
    pd.DataFrame({'wind_speed_ms': [0.0, v_cutin - 0.01], 'power_kW': [0.0, 0.0]}),
    df_pc,
    pd.DataFrame({'wind_speed_ms': [v_cutout + 0.01], 'power_kW': [0.0]})
], ignore_index=True).sort_values('wind_speed_ms').reset_index(drop=True)

# Build interpolator
pc_interp = interp1d(
    df_pc['wind_speed_ms'], df_pc['power_kW'],
    kind='linear', bounds_error=False, fill_value=0.0
)

print(f"Turbine      : {description}")
print(f"Rotor diam   : {rotor_diameter} m")
print(f"Hub height   : 103.6 m (from WAsP SuggestedHeights)")
print(f"Cut-in/out   : {v_cutin} / {v_cutout} m/s")
print(f"Rated power  : {df_pc['power_kW'].max():,.0f} kW  ({df_pc['power_kW'].max()/1000:.0f} MW)")
print(f"Reference rho: {rho_ref} kg/m³")

# Plot power curve
fig, ax = plt.subplots(figsize=(8, 4))
ws_plot = np.linspace(0, 27, 300)
ax.plot(ws_plot, pc_interp(ws_plot) / 1000, lw=2, color='#1976D2')
ax.scatter(df_pc['wind_speed_ms'], df_pc['power_kW'] / 1000,
           s=20, color='#1976D2', zorder=5)
ax.set_xlabel("Wind speed [m/s]")
ax.set_ylabel("Power [MW]")
ax.set_title(f"V164-8MW power curve  (rated: 8 MW, cut-in: {v_cutin} m/s, cut-out: {v_cutout} m/s)")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# =============================================================
# 2. LOAD ERA5 WIND SPEED CSV
# =============================================================
ws_path = os.path.join(BASE_DIR, "..", "data", "windspeeds.csv")

# Skip the metadata header rows (lat, lon, elevation, etc.)
# Find the row where 'time' appears
with open(ws_path, 'r') as f:
    lines = f.readlines()

header_row = next(i for i, l in enumerate(lines) if l.startswith('time'))

df_era5 = pd.read_csv(
    ws_path,
    skiprows=header_row,
    parse_dates=['time']
)
df_era5.columns = ['time', 'wind_speed_ms']
df_era5.set_index('time', inplace=True)
df_era5.index = pd.to_datetime(df_era5.index)

# ERA5 is in UTC (GMT+0 per your file) — convert to Danish local time
df_era5.index = df_era5.index.tz_localize('UTC').tz_convert('Europe/Copenhagen')
df_era5.index = df_era5.index.tz_localize(None)  # strip tz for easy merging

print(f"\nERA5 records : {len(df_era5)} hourly")
print(f"Date range   : {df_era5.index[0]}  →  {df_era5.index[-1]}")
print(f"Mean wind    : {df_era5['wind_speed_ms'].mean():.2f} m/s")
print(f"Max wind     : {df_era5['wind_speed_ms'].max():.2f} m/s")


# =============================================================
# 3. HUB HEIGHT CORRECTION: 100m → 103.6m
# =============================================================
Z_ERA5   = 100.0   # ERA5 wind height [m]
Z_HUB    = 103.6   # V164 hub height from WAsP file [m]
Z0_SEA   = 0.0002  # roughness length for open sea [m]

def log_law(v_ref, z_ref, z_target, z0=Z0_SEA):
    return v_ref * np.log(z_target / z0) / np.log(z_ref / z0)

df_era5['wind_speed_hub'] = log_law(
    df_era5['wind_speed_ms'].values, Z_ERA5, Z_HUB
)
correction_factor = np.log(Z_HUB / Z0_SEA) / np.log(Z_ERA5 / Z0_SEA)
print(f"\nHub height correction factor (100m → 103.6m): {correction_factor:.4f}")
# Will be ~1.003 — negligible but correct


# =============================================================
# 4. APPLY POWER CURVE + SCALE TO 210 MW FARM
# =============================================================
FARM_CAPACITY_KW  = 210_000.0    # kW
TURBINE_RATED_KW  = 8_000.0      # kW  (V164-8MW)
N_TURBINES        = FARM_CAPACITY_KW / TURBINE_RATED_KW   # = 26.25 turbines

# Single turbine power
df_era5['power_1turbine_kW'] = pc_interp(df_era5['wind_speed_hub'].values)

# Scaled to 210 MW farm
df_era5['wind_power_kW'] = df_era5['power_1turbine_kW'] * N_TURBINES

cf_era5 = df_era5['wind_power_kW'].mean() / FARM_CAPACITY_KW * 100
print(f"\nERA5-derived capacity factor : {cf_era5:.1f}%")
print(f"Number of equivalent turbines: {N_TURBINES:.2f}")


# =============================================================
# 5. UPSAMPLE ERA5 TO 15-MINUTE
# =============================================================
# Remove duplicate timestamps (can occur from timezone handling)
df_era5_clean = df_era5[~df_era5.index.duplicated(keep='first')]

#df_era5_15min = df_era5_clean[['wind_power_kW']].resample('15min').interpolate(method='linear')

df_era5_15min = df_era5_clean[['wind_speed_hub', 'wind_power_kW']].resample('15min').interpolate(method='linear')

print(f"\nAfter upsampling: {len(df_era5_15min)} records at 15-min resolution")

neg_da_idx = df_analysis[df_analysis['da_price'] < 0].index
neg_imb_idx = df_analysis[df_analysis['imb_price'] < 0].index

mean_wind_neg_da = df_era5_15min.loc[df_era5_15min.index.intersection(neg_da_idx), 'wind_speed_hub'].mean()
mean_wind_neg_imb = df_era5_15min.loc[df_era5_15min.index.intersection(neg_imb_idx), 'wind_speed_hub'].mean()

print(f"Mean ERA5 hub-height wind during DA price < 0   : {mean_wind_neg_da:.2f} m/s")
print(f"Mean ERA5 hub-height wind during Imbalance < 0 : {mean_wind_neg_imb:.2f} m/s")

# =============================================================
# 6. COMPARE ERA5 vs SCADA
# =============================================================
# Align on common index
common_idx  = df_era5_15min.index.intersection(wind_df.index)
era5_arr    = df_era5_15min.loc[common_idx, 'wind_power_kW'].values
scada_arr   = wind_df.loc[common_idx, 'wind_kW'].values

corr = np.corrcoef(era5_arr, scada_arr)[0, 1]
mae  = np.abs(era5_arr - scada_arr).mean()
rmse = np.sqrt(((era5_arr - scada_arr) ** 2).mean())
bias = (era5_arr - scada_arr).mean()
cf_scada = scada_arr.mean() / FARM_CAPACITY_KW * 100

print("\n=== ERA5 vs SCADA COMPARISON ===")
print(f"  Common timesteps  : {len(common_idx)}")
print(f"  Correlation       : {corr:.3f}")
print(f"  MAE               : {mae:,.0f} kW  ({mae/FARM_CAPACITY_KW*100:.1f}% of rated)")
print(f"  RMSE              : {rmse:,.0f} kW  ({rmse/FARM_CAPACITY_KW*100:.1f}% of rated)")
print(f"  Mean bias         : {bias:+,.0f} kW  ({'ERA5 higher' if bias > 0 else 'ERA5 lower'})")
print(f"  SCADA capacity factor : {cf_scada:.1f}%")
print(f"  ERA5  capacity factor : {cf_era5:.1f}%")


# =============================================================
# 7. PLOTS
# =============================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("ERA5 wind power vs SCADA — V164-8MW power curve", fontsize=13, fontweight='bold')

# 7a. One-week time series
n_week = 96 * 7
idx_week = common_idx[:n_week]
axes[0, 0].plot(era5_arr[:n_week] / 1000,  lw=1,   color='#E53935', label='ERA5', alpha=0.9)
axes[0, 0].plot(scada_arr[:n_week] / 1000, lw=1,   color='#1976D2', label='SCADA', alpha=0.9)
axes[0, 0].set_ylabel("Power [MW]")
axes[0, 0].set_title("First week: ERA5 vs SCADA")
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 7b. Scatter
axes[0, 1].scatter(scada_arr / 1000, era5_arr / 1000,
                   s=1, alpha=0.08, color='#388E3C')
lim = FARM_CAPACITY_KW / 1000 * 1.05
axes[0, 1].plot([0, lim], [0, lim], 'k--', lw=0.8, label='1:1')
axes[0, 1].set_xlabel("SCADA power [MW]")
axes[0, 1].set_ylabel("ERA5 power [MW]")
axes[0, 1].set_title(f"Scatter  (corr={corr:.3f}, bias={bias/1000:+.1f} MW)")
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(alpha=0.3)

# 7c. Difference distribution
diff = era5_arr - scada_arr
axes[1, 0].hist(diff / 1000, bins=100, color='#7B1FA2', alpha=0.7, edgecolor='none')
axes[1, 0].axvline(0,            color='black', lw=0.8, ls='--')
axes[1, 0].axvline(bias / 1000,  color='#7B1FA2', lw=1.5, ls='-',
                   label=f'Mean bias: {bias/1000:+.1f} MW')
axes[1, 0].set_xlabel("ERA5 − SCADA [MW]")
axes[1, 0].set_ylabel("Count")
axes[1, 0].set_title("Difference distribution")
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# 7d. Monthly capacity factor comparison
df_compare = pd.DataFrame({
    'era5_kW':  era5_arr,
    'scada_kW': scada_arr,
}, index=common_idx)
df_compare['month'] = df_compare.index.month

month_names = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']
months = sorted(df_compare['month'].unique())
cf_era5_m  = [df_compare.loc[df_compare['month']==m, 'era5_kW'].mean()
               / FARM_CAPACITY_KW * 100 for m in months]
cf_scada_m = [df_compare.loc[df_compare['month']==m, 'scada_kW'].mean()
               / FARM_CAPACITY_KW * 100 for m in months]

x = np.arange(len(months))
w = 0.35
axes[1, 1].bar(x - w/2, cf_era5_m,  width=w, label='ERA5',  color='#E53935', alpha=0.8)
axes[1, 1].bar(x + w/2, cf_scada_m, width=w, label='SCADA', color='#1976D2', alpha=0.8)
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels([month_names[m-1] for m in months])
axes[1, 1].set_ylabel("Capacity factor [%]")
axes[1, 1].set_title("Monthly capacity factor: ERA5 vs SCADA")
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()