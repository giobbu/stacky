import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import genextreme as gev

class EVTAnalyzer:
    " Extreme Value Theory Analyzer"
    def __init__(self, data, method='GEV'):
        assert method in ['GEV', 'GPD', 'EMP'], 'Method must'
        assert isinstance(data, pd.Series), 'Data must be a pandas Series'
        self.data = data
        self.method = method

    def _fit_GEV(self):
        " Fit GEV distribution to data"
        shape, loc, scale = gev.fit(self.data.values, 0)
        return shape, loc, scale
    
    def _quantile_GEV(self, value, shape, loc, scale):
        " Calculate quantile for given value"
        quantile = gev.cdf(value, shape, loc, scale)
        return quantile
    
    def generate_sample(self, size=10000):
        " Generate sample from fitted distribution"
        shape, loc, scale = self._fit_GEV()
        samples = gev.rvs(shape, loc, scale, size=size)
        return samples
    
    def estimate_return_level(self, quantile, loc, scale, shape):
        "Estimate return level for given quantile"
        shape, loc, scale = self._fit_GEV()
        level = loc +  scale/ shape * (1 - (-np.log(quantile)) ** (shape))
        return level

    def get_return_level(self, return_period):
        "Estimate return level for given return period"
        quantile = 1 - 1/return_period
        return_level = self.estimate_return_level(quantile)
        return return_level

    def get_return_period(self, return_level):
        "Estimate return period for given return level"
        shape, loc, scale = self._fit_GEV()
        quantile = self._quantile_GEV(return_level, shape, loc, scale)
        return_period = 1/(1-quantile)
        return return_period
    
if __name__ == '__main__':

    # Generate some random data
    data = np.random.normal(0, 50, 10000)
    data = pd.Series(data)
    # Block maximas
    block_maxima = data.rolling(window=10).max().dropna()

    evt = EVTAnalyzer(block_maxima)
    shape, loc, scale = evt._fit_GEV()
    value = 150
    quantile = evt._quantile_GEV(value, shape, loc, scale)
    return_level = evt.estimate_return_level(quantile, loc, scale, shape)
    return_period = evt.get_return_period(return_level)

    print('return_period', return_period)
    print('return_level', return_level)
    print('quantile', quantile)

    import matplotlib.pyplot as plt
    # subplots horizontal
    figure = plt.figure(figsize=(10, 5))
    plt.hist(data, bins=20, density=True, label='data', alpha=0.5)
    plt.hist(block_maxima, bins=20, density=True, label='block_maxima')
    x_r80 = np.arange(-100, 500)
    plt.plot(x_r80, gev.pdf(x_r80, shape, loc=loc, scale=scale), "k", lw=3, label='GEV')
    plt.plot(x_r80, norm.pdf(x_r80, loc=loc, scale=scale), "r", lw=1, label='Normal', alpha=0.5)
    plt.axvline(value, color='red', label='value')
    plt.legend()
    plt.title(f'Extreme Value Theory - GEV Distribution - Quantile: {quantile:.2f} - Return Level (Set): {return_level:.2f} - Return Period: {return_period:.2f}')
    plt.show()

