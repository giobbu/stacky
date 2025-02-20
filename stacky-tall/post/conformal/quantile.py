import numpy as np

def conformalized_qr(n, cal_y, cal_X, val_lower, val_upper, model_lower, model_upper, alpha):
    """ Conformal Prediction for Quantile Regression.
    ## Reference: https://people.eecs.berkeley.edu/~angelopoulos/publications/downloads/gentle_intro_conformal_dfuq.pdf
    """
    # Get scores
    cal_scores = np.maximum(cal_y - model_upper(cal_X), model_lower(cal_X) - cal_y)
    # Get the score quantile
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, interpolation='higher')
    # Deploy (output=lower and upper adjusted quantiles)
    prediction_sets = [val_lower - qhat, val_upper + qhat]
    return prediction_sets