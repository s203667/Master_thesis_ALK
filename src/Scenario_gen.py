import numpy as np

def generate_scenarios(price_fc_day, wind_fc_day, imbalance_fc_day,
                    hist_price_errors, hist_wind_errors, hist_imb_errors,
                    hist_wind_forecasts,   # NEW: (N_days, 96) forecasts
                    n_scenarios=20, n_neighbors=50):
    """
    Generate joint scenarios by sampling historical forecast error days.

    Parameters
    ----------
    price_fc_day    : (96,) array  - lag-1 price forecast for today
    wind_fc_day     : (96,) array  - lag-1 wind forecast for today
    imbalance_fc_day: (96,) array  - lag-1 imbalance price forecast for today
    hist_price_errors: (N_days, 96) - historical (actual - forecast) for price
    hist_wind_errors : (N_days, 96) - historical (actual - forecast) for wind
    hist_imb_errors  : (N_days, 96) - historical (actual - forecast) for imbalance
    n_scenarios     : int           - number of scenarios S
    seed            : int or None

    Returns
    -------
    price_sc  : (S, 96) array
    wind_sc   : (S, 96) array
    imb_sc    : (S, 96) array
    probs     : (S,)    array  - uniform 1/S
    """

    """
    Sample scenarios from the n_neighbors historical days whose wind forecast
    most closely resembles today's wind forecast.
    """


    # Similarity: Euclidean distance between today's forecast and each historical day's forecast
    distances = np.linalg.norm(hist_wind_forecasts - wind_fc_day[None, :], axis=1)
    
    # Pick the n_neighbors closest days
    neighbor_idx = np.argsort(distances)[:n_neighbors]
    
    # Sample n_scenarios from those neighbors
    rng = np.random.default_rng()
    sampled = rng.choice(neighbor_idx, size=n_scenarios, replace=True)
    

    price_sc = price_fc_day[None, :] + hist_price_errors[sampled]
    wind_sc  = wind_fc_day[None,  :] + hist_wind_errors[sampled]
    imb_sc   = imbalance_fc_day[None, :] + hist_imb_errors[sampled]

    
    wind_sc  = np.clip(wind_sc, 0, None)
    
    probs = np.ones(n_scenarios) / n_scenarios
    return price_sc, wind_sc, imb_sc, probs


def compute_forecast_errors(price_real, wind_real, imbalance_real, lag=96):
    T = len(price_real)
    n_days = T // lag - 1
    
    price_errors   = np.zeros((n_days, lag))
    wind_errors    = np.zeros((n_days, lag))
    imb_errors     = np.zeros((n_days, lag))
    wind_forecasts = np.zeros((n_days, lag))   # NEW
    
    for d in range(n_days):
        start  = (d + 1) * lag
        end    = (d + 2) * lag
        f_start = d * lag
        f_end   = (d + 1) * lag
        
        price_errors[d]   = price_real[start:end]  - price_real[f_start:f_end]
        wind_errors[d]    = wind_real[start:end]   - wind_real[f_start:f_end]
        imb_errors[d]     = imbalance_real[start:end] - imbalance_real[f_start:f_end]
        wind_forecasts[d] = wind_real[f_start:f_end]   # NEW: the forecast for that day
    
    return price_errors, wind_errors, imb_errors, wind_forecasts