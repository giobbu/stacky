import numpy as np

def crps_ensemble_vectorized(observations, forecasts, weights=1):
    " Compute the Continuous Ranked Probability Score (CRPS) for an ensemble forecast."
    # Taken from: https://github.com/properscoring/properscoring/blob/master/properscoring/_crps.py#L244
    
    # Continuous Ranked Probability Score (CRPS) formula:
    # 
    #            CRPS(F, x) = E_F |X - x| - (1/2) E_F |X - X'|
    # 
    # where:
    # - F represents the forecast distribution.
    # - X and X' are independent random variables drawn from F.
    # - E_F denotes the expectation under F.
    # - The first term measures the average absolute error between forecasts and the observation.
    # - The second term adjusts for the internal spread of the ensemble forecasts.

    observations = np.asarray(observations)
    forecasts = np.asarray(forecasts)
    weights = np.asarray(weights)
    
    if weights.ndim > 0:
        weights = np.where(~np.isnan(forecasts), weights, np.nan)
        weights = weights / np.nanmean(weights, axis=-1, keepdims=True)

    if observations.ndim == forecasts.ndim - 1:
        assert observations.shape == forecasts.shape[:-1]
        observations = observations[..., np.newaxis]
        score = np.nanmean(weights * abs(forecasts - observations), -1)

        forecasts_diff = (np.expand_dims(forecasts, -1) -
                            np.expand_dims(forecasts, -2))
        weights_matrix = (np.expand_dims(weights, -1) *
                            np.expand_dims(weights, -2))
        
        score += -0.5 * np.nanmean(weights_matrix * abs(forecasts_diff),
                                    axis=(-2, -1))
        return score
    elif observations.ndim == forecasts.ndim:
        return abs(observations - forecasts)

# Example inputs
observations = np.array(3.5)
forecasts = np.array([1, 2, 3, 4])
weights = np.ones_like(forecasts)

# Compute CRPS
crps_score = crps_ensemble_vectorized(observations, forecasts, weights)
print("CRPS Score:", crps_score)