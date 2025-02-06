import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from scipy.stats import genextreme as gev

class GEVAnalyzer:
    "Extreme Value Theory Analyzer"
    def __init__(self, data):
        assert isinstance(data, pd.Series), 'Data must be a pandas Series'
        self.data = data

    def compute_block_maxima(self, period_window=10):
        " Compute block maxima"
        self.block_maxima = self.data.rolling(window=period_window).max().dropna()
        return self.block_maxima

    def maximum_likelihood_estimation(self):
        " Fit GEV distribution to data"
        shape, loc, scale = gev.fit(self.block_maxima.values, 0)
        return shape, loc, scale
    
    def cdf_GEV(self, value, shape, loc, scale):
        " Calculate probability for given value"
        probability = gev.cdf(value, shape, loc, scale)
        return probability
    
    def _estimate_return_level(self, probability, loc, scale, shape):
        "Estimate return level for given probability"
        shape, loc, scale = self.maximum_likelihood_estimation()
        return_level = gev.ppf(probability, shape, loc, scale)
        return return_level
    
    def get_empirical_return_period(self):
        """
        Compute empirical return priod for observed data
        """
        df = pd.DataFrame(index=np.arange(self.block_maxima.size))
        # sort the data
        df["sorted"] = np.sort(self.block_maxima)[::-1]
        # rank via scipy instead to deal with duplicate values
        df["ranks_sp"] = np.sort(stats.rankdata(-self.block_maxima))
        # find exceedence probability
        n = self.block_maxima.size
        df["exceedance"] = df["ranks_sp"] / (n + 1)
        # find return period
        df["period"] = 1 / df["exceedance"]
        return df

    def get_return_level(self, return_period):
        "Estimate return level for given return period"
        probability = 1 - 1/return_period
        return_level = self._estimate_return_level(probability)
        return return_level

    def get_return_period(self, level_value):
        "Estimate return period for given return level"
        shape, loc, scale = self.maximum_likelihood_estimation()
        probability = self.cdf_GEV(level_value, shape, loc, scale)
        return_period = 1/(1-probability)
        return return_period
    
if __name__ == '__main__':

    # Generate some random data
    data = np.random.normal(0, 50, 10000)
    data = pd.Series(data)

    # Initialize GEVAnalyzer
    evt = GEVAnalyzer(data)

    # Block maximas
    block_maxima = evt.compute_block_maxima(period_window=10)

    # Compute maximum likelihood estimation
    shape, loc, scale = evt.maximum_likelihood_estimation()

    # Set return period
    value = 150

    # Calculate probability
    probability = evt.cdf_GEV(value, shape, loc, scale)

    # Calculate return level
    return_level = evt._estimate_return_level(probability, loc, scale, shape)

    # Calculate return period
    return_period = evt.get_return_period(return_level)

    print('return_period', return_period)
    print('return_level', return_level)
    print('probability of exceedance', 1 - probability)
    print('probability', probability)

    import matplotlib.pyplot as plt
    # subplots horizontal
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f'GEV - Probability of Exceedance: {1 - probability:.2f} - Return Level (Set): {return_level:.2f} - Return Period: {return_period:.2f}')

    ax[0].hist(data, bins=20, density=True, label='all data', alpha=0.5)
    ax[0].hist(block_maxima, bins=20, density=True, label='block maxima data')
    x_r80 = np.arange(-100, 500)
    ax[0].plot(x_r80, gev.pdf(x_r80, shape, loc=loc, scale=scale), "k", lw=3, label='GEV (good fit)')
    ax[0].plot(x_r80, norm.pdf(x_r80, loc=loc, scale=scale), "r", lw=1, label='Normal (bad fit)', alpha=0.5)
    ax[0].axvline(value, color='b', label='return level', linestyle='--')
    ax[0].legend()

    # plot cdf
    x = np.linspace(-100, 500, 1000)
    y = gev.cdf(x, shape, loc=loc, scale=scale)
    ax[1].plot(x, y)
    ax[1].axhline(probability, color='b', linestyle='--')
    ax[1].axvline(value, color='b', linestyle='--')
    ax[1].legend()
    # save plot
    plt.savefig('imgs/gev.png')
    plt.show()

