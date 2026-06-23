import numpy as np
import os

def compute_forecast_errors_cached(
    price_real, wind_real, imbalance_real,
    lag=96,
    cache_dir="cache",
    cache_file="forecast_errors.npz",
):
    """
    Compute historical forecast errors once and cache them.
    These come from the full dataset and never change between runs.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, cache_file)

    if os.path.exists(cache_path):
        d = np.load(cache_path)
        print(f"Loaded forecast errors from cache ({cache_path})")
        return d["price_errors"], d["wind_errors"], d["imb_errors"], d["wind_forecasts"]

    price_errors, wind_errors, imb_errors, wind_forecasts = compute_forecast_errors(
        price_real, wind_real, imbalance_real, lag=lag
    )

    np.savez(cache_path,
             price_errors=price_errors,
             wind_errors=wind_errors,
             imb_errors=imb_errors,
             wind_forecasts=wind_forecasts)
    print(f"Computed and cached forecast errors → {cache_path}")

    return price_errors, wind_errors, imb_errors, wind_forecasts
# ─────────────────────────────────────────────────────────────────────────────
# Forecast error computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_forecast_errors(price_real, wind_real, imbalance_real, lag=96):
    """
    Compute lag-1 forecast errors for each full day in the time series.

    Returns
    -------
    price_errors   : (N_days, lag)  actual - forecast for DA price
    wind_errors    : (N_days, lag)  actual - forecast for wind
    imb_errors     : (N_days, lag)  actual - forecast for imbalance price
    wind_forecasts : (N_days, lag)  the lag-1 wind forecast for each day
                                    (used by the 'similar' sampling mode)
    """
    T      = len(price_real)
    n_days = T // lag - 1

    price_errors   = np.zeros((n_days, lag))
    wind_errors    = np.zeros((n_days, lag))
    imb_errors     = np.zeros((n_days, lag))
    wind_forecasts = np.zeros((n_days, lag))

    for d in range(n_days):
        start   = (d + 1) * lag
        end     = (d + 2) * lag
        f_start = d * lag
        f_end   = (d + 1) * lag

        price_errors[d]   = price_real[start:end]     - price_real[f_start:f_end]
        wind_errors[d]    = wind_real[start:end]      - wind_real[f_start:f_end]
        imb_errors[d]     = imbalance_real[start:end] - imbalance_real[f_start:f_end]
        wind_forecasts[d] = wind_real[f_start:f_end]

    return price_errors, wind_errors, imb_errors, wind_forecasts


# ─────────────────────────────────────────────────────────────────────────────
# Scenario generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_scenarios(
    price_fc_day,
    wind_fc_day,
    imbalance_fc_day,
    hist_price_errors,
    hist_wind_errors,
    hist_imb_errors,
    hist_wind_forecasts,
    n_scenarios=20,
    n_neighbors=50,
    mode="random",
    seed=None,
):
    """
    Generate joint scenarios by sampling historical forecast error days.

    Parameters
    ----------
    price_fc_day     : (96,)        lag-1 price forecast for today
    wind_fc_day      : (96,)        lag-1 wind forecast for today
    imbalance_fc_day : (96,)        lag-1 imbalance price forecast for today
    hist_price_errors : (N, 96)     historical (actual - forecast) for price
    hist_wind_errors  : (N, 96)     historical (actual - forecast) for wind
    hist_imb_errors   : (N, 96)     historical (actual - forecast) for imbalance
    hist_wind_forecasts : (N, 96)   lag-1 wind forecast for each historical day
                                    (only used when mode='similar' or 'adversarial')
    n_scenarios : int               number of scenarios S
    n_neighbors : int               pool size for 'similar' and 'adversarial' modes
    mode : str
        'random'     — sample uniformly from ALL historical error days.
                       Unbiased; recommended for the base stochastic model.
                       Represents the full empirical error distribution.

        'similar'    — sample from the n_neighbors days whose wind forecast
                       most closely matches today's (Euclidean distance).
                       Exploits pattern similarity; may underestimate tail risk.

        'adversarial'— sample from the n_neighbors days whose wind forecast
                       is MOST DIFFERENT from today's.
                       Use for robustness / worst-case sensitivity analysis only.
                       Will produce overly conservative DA decisions if used
                       as the base case.

    seed : int or None              random seed for reproducibility

    Returns
    -------
    price_sc : (S, 96)   scenario price realisations
    wind_sc  : (S, 96)   scenario wind realisations  (clipped to >= 0)
    imb_sc   : (S, 96)   scenario imbalance price realisations
    probs    : (S,)      equal probability weights 1/S
    """
    rng = np.random.default_rng(seed)
    N   = len(hist_price_errors)

    if mode == "random":
        # ── Standard approach: uniform sample from all historical error days ──
        # Recommended for the base stochastic model.
        # Each historical day is equally likely, preserving the full empirical
        # joint distribution of (price error, wind error, imbalance error).
        sampled = rng.choice(N, size=n_scenarios, replace=True)

    elif mode == "similar":
        # ── Sample from days with similar wind forecast (original approach) ──
        distances   = np.linalg.norm(
            hist_wind_forecasts - wind_fc_day[None, :], axis=1
        )
        neighbor_idx = np.argsort(distances)[:n_neighbors]
        sampled      = rng.choice(neighbor_idx, size=n_scenarios, replace=True)

    elif mode == "adversarial":
        # ── Sample from days MOST DIFFERENT from today's wind forecast ────────
        # Use only for robustness / sensitivity analysis.
        # Produces worst-case scenarios; will bias DA decisions toward
        # over-hedging if used as the base case.
        distances    = np.linalg.norm(
            hist_wind_forecasts - wind_fc_day[None, :], axis=1
        )
        neighbor_idx = np.argsort(distances)[-n_neighbors:]   # furthest
        sampled      = rng.choice(neighbor_idx, size=n_scenarios, replace=True)

    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Choose 'random', 'similar', or 'adversarial'."
        )

    # ── Add sampled errors to today's forecast ────────────────────────────────
    price_sc = price_fc_day[None, :]      + hist_price_errors[sampled]
    wind_sc  = wind_fc_day[None, :]       + hist_wind_errors[sampled]
    imb_sc   = imbalance_fc_day[None, :]  + hist_imb_errors[sampled]

    wind_sc = np.clip(wind_sc, 0, None)   # wind cannot be negative

    probs = np.ones(n_scenarios) / n_scenarios
    return price_sc, wind_sc, imb_sc, probs