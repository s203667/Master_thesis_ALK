# Rolling Horizon & Forecasting Explanation

## Overview
Your system uses a **receding horizon optimization** (Model Predictive Control-like) approach:
- Solve a 96-step optimization window (H=96, ~1 day)
- Implement only the first 24 steps (step=24, ~6 hours)
- Slide forward 24 steps and repeat
- Carry over state (storage, on/off, power) to next window

---

## Step-by-Step Process

### 1. **Setup Phase** (lines 47-96 in main.py)

```python
H = 96           # Planning horizon = 1 day (96 × 15min steps)
step = 24        # Implementation window = 6 hours (24 × 15min steps)
lag = 96         # Forecast lag = 1 day (use previous day as forecast)
```

**Key data:**
- `wind_real`, `price_real`: actual historical data
- `wind_forecast`, `price_forecast`: lagged versions (day before rolled forward)

**FCR data:**
- Indexed by 4-hour blocks: `i0 = t0 // 16` (each block = 16 timesteps)
- Must map correctly between 15-min timesteps and 4-hour blocks

---

### 2. **Rolling Horizon Loop** (lines 162 onwards)

Each iteration processes one "window":

#### Window 1: t0 = 0
```
|------- Planning Horizon (H=96) --------|
|--- Implement (step=24) ---|
wind_forecast[0:96]        ← Full 96-step forecast
price_forecast[0:96]       ← Full 96-step forecast
Optimize over 96 steps
Execute only first 24 steps
```

#### Window 2: t0 = 24
```
                      |------- Planning Horizon (H=96) --------|
                      |--- Implement (step=24) ---|
wind_forecast[24:120]      ← Next 96-step forecast (rolling window)
price_forecast[24:120]     ← Next 96-step forecast
Optimize over 96 steps
Execute only first 24 steps
```

---

### 3. **Forecasting Mechanism** (lines 88-95)

```python
lag = 96  # 1 day lag
wind_forecast = np.roll(wind_real, lag)
price_forecast = np.roll(price_real, lag)

# For first day, use actual values
wind_forecast[:lag] = wind_real[:lag]
price_forecast[:lag] = price_real[:lag]
```

**Example:**
```
Real wind:     [t0, t1, t2, ..., t95, t96, t97, ...]
Forecast lag:   [t0, t1, ..., t95, t0,  t1,  ...]  (shifted by 96)
                                   ↑
                            This is the forecast for t96
                            (uses historical t0 data)
```

So at each horizon, you're optimizing with:
- **Known**: Current storage, on/off state, power
- **Uncertain**: Next 96 steps of wind and price (from day-before pattern)
- **Constraint**: Only implement next 24 steps; then replan

---

### 4. **Data Slicing per Window** (lines 169-187)

For window starting at `t0`:

```python
# Real data used (what actually happened)
wind_slice_real = wind_real[t0:t0 + step_eff]        # 24 values
price_slice_real = price_real[t0:t0 + step_eff]      # 24 values

# Forecast data used (what we think will happen)
wind_slice_forecast = wind_forecast[t0:t0 + H_eff]   # 96 values
price_slice_forecast = price_forecast[t0:t0 + H_eff] # 96 values

# FCR blocks (map 15-min to 4-hour blocks)
i0 = t0 // 16                           # Block index
I_eff = int(np.ceil(H_eff / 16))       # Number of blocks
price_FCR_slice = price_FCR[i0:i0 + I_eff]
FCR_volume_slice = FCR_volume[i0:i0 + I_eff]
```

---

### 5. **State Carryover** (lines 189-228)

After solving, extract state for next window:

```python
# From previous window
storage_state = 0.0         # kg
z_on_state = 1              # 0 or 1
P_prev = 0                  # kW

# After solving window
storage_state = model.results.storage[step_eff - 1]  # Storage at end of implemented part
z_on_state = model.results.z_on[step_eff - 1]       # Was it on at end?
P_prev = sum(model.model.P_seg[step_eff - 1, s].value for s in model.model.S)  # Total power at end
```

These are passed to `InputData` for next window:
```python
storage_initial=storage_state,
z_on_prev=z_on_state,
P_prev=P_prev
```

This ensures:
- ✓ Continuity: Storage level doesn't jump
- ✓ Causality: Power ramps smoothly
- ✓ Realism: Can't turn off if still ramping down

---

### 6. **Results Reconstruction** (lines 300-330)

Each window produces results for H_eff steps, but only the first `step_eff` are "true" decisions:

```python
results_all = [
    {h, storage, z_on, P_buy, P_sell, r_FCR_block, wind_real, price_real, ...},
    {h, storage, z_on, P_buy, P_sell, r_FCR_block, wind_real, price_real, ...},
    ...
]
```

Reconstruct by taking only implemented part of each:
```python
for i, result in enumerate(results_all):
    if i == last_window:
        take_all(result)   # Last window: take all (may be partial)
    else:
        take_first_step_eff(result)  # Earlier windows: only first 24 steps
```
 
This avoids double-counting decisions that appear in multiple overlapping windows.

---

### 7. **FCR Block Expansion** (line 324)

```python
# Each FCR decision spans 4 hours = 16 timesteps
for block_value in result["r_FCR_block"]:
    all_fcr_blocks.extend([block_value] * 16)
```

So if `r_FCR_block = [100, 150, 120]` (3 blocks, 4 hours each = 12 hours):
```
15-min timesteps: [100, 100, 100, 100,  150, 150, 150, 150,  120, 120, 120, 120]
                   (4 hours)             (4 hours)             (4 hours)
```

---

## Key Considerations

| Aspect | Details |
|--------|---------|
| **Forecast quality** | Only yesterday's data → assumes similar patterns. Real systems use weather models. |
| **Planning vs. execution** | Plan 96 steps ahead, execute 24. Others plans in window 2 override window 1's steps 25+ |
| **State continuity** | Storage, power ramps must be smooth. `P_prev` enforces this. |
| **FCR blocks** | Must map 15-min indices to 4-hour blocks correctly. |
| **Edge cases** | At end of dataset: last window may be < H steps. Handled by `H_eff = min(H, len(wind) - t0)` |

---

## Testing Checklist

1. **Forecast correctness**: Is `wind_forecast[t]` really `wind_real[t-96]`?
2. **State carryover**: Does storage/power smoothly transition between windows?
3. **Data slicing**: Are slices correct lengths? No off-by-one errors?
4. **FCR mapping**: Do 4-hour blocks align with 15-min timesteps?
5. **Reconstruction**: Full timeline length = original data length? No duplicates or gaps?
6. **Edge cases**: Handles last partial window? Works for short datasets?

