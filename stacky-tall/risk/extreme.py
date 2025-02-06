import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from scipy.stats import genextreme as gev
# joblib is used for parallel processing
from joblib import Parallel, delayed

class GEVAnalyzer:
    "Extreme Value Theory Analyzer"

    def __init__(self, data):
        assert isinstance(data, pd.Series), 'Data must be a pandas Series'
        self.data = data

    def compute_block_maxima(self, period_window=10):
        " Compute block maxima"
        self.block_maxima = self.data.rolling(window=period_window).max().dropna()
        return self.block_maxima
    
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

    def maximum_likelihood_estimation(self):
        " Fit GEV distribution to data"
        shape, loc, scale = gev.fit(self.block_maxima.values, 0)
        return shape, loc, scale
    
    def cdf_GEV(self, value, shape, loc, scale):
        " Calculate probability for given value"
        probability = gev.cdf(value, shape, loc, scale)
        return probability
    
    def _bootstrap_samples(self, n_samples):
        "Resample with replacement"
        n = self.block_maxima.size
        list_of_samples = []
        for i in range(n_samples):
            sample = np.random.choice(self.block_maxima, n)
            list_of_samples.append(sample)
        return list_of_samples
    
    def _estimate_return_level(self, probability, loc, scale, shape):
        "Estimate return level for given probability"
        return_level = gev.ppf(probability, shape, loc, scale)
        return return_level
    
    def _bootstrap_params(self, n_samples=100):
        "Estimate parameters for bootstrap samples"
        bootstrap_params = Parallel(n_jobs=-1)(delayed(gev.fit)(sample, 0) for sample in self._bootstrap_samples(n_samples))
        return bootstrap_params
    
    def get_return_level(self, return_period, with_confidence=False):
        "Estimate return level for given return period"
        probability = 1 - 1/return_period
        shape, loc, scale = self.maximum_likelihood_estimation()
        return_level = self._estimate_return_level(probability, loc, scale, shape)
        if with_confidence:
            bootstrap_return_levels = [ self._estimate_return_level(probability, loc, scale, shape) for shape, loc, scale in self._bootstrap_params()]
            lower, upper = np.percentile(bootstrap_return_levels, [2.5, 97.5])
            return return_level, lower, upper
        return return_level

    def get_return_period(self, return_level, with_confidence=False):
        "Estimate return period for given return level"
        shape, loc, scale = self.maximum_likelihood_estimation()
        return_period = 1/(1-self.cdf_GEV(return_level, shape, loc, scale))
        if with_confidence:
            bootstrap_return_periods = [1/(1-self.cdf_GEV(return_level, shape, loc, scale)) for shape, loc, scale in self._bootstrap_params()]
            lower, upper = np.percentile(bootstrap_return_periods, [2.5, 97.5])
            return return_period, lower, upper
        return return_period



if __name__ == '__main__':

    # Generate some random data
    data = pd.Series(np.random.normal(0, 50, 1000))

    # Initialize GEVAnalyzer
    evt = GEVAnalyzer(data)

    # Block maximas
    block_maxima = evt.compute_block_maxima(period_window=10)

    # Compute maximum likelihood estimation
    shape, loc, scale = evt.maximum_likelihood_estimation()

    # Set return period
    set_return_period = 10

    # Probability 
    probability = 1 - 1/set_return_period

    # Get return level with confidence interval
    return_level, lower_return_level, upper_return_level = evt.get_return_level(set_return_period, with_confidence=True)

    # Get return period with confidence interval
    return_period, lower_return_period, upper_return_period = evt.get_return_period(return_level, with_confidence=True)

    print(f" With {return_period}-Year Return Period:")
    print(f"Return Level: {return_level:.2f} - Lower: {lower_return_level:.2f} - Upper: {upper_return_level:.2f}")


    import matplotlib.pyplot as plt
    # subplots horizontal
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'GEV Analysis for {set_return_period}-Year Return Period: Return Level = {return_level:.2f} - Lower: {lower_return_level:.2f} - Upper: {upper_return_level:.2f}')

    # plot pdf
    ax[0].hist(data, bins=20, density=True, label='all data', alpha=0.5)
    ax[0].hist(block_maxima, bins=20, density=True, label='block maxima data')
    x_r80 = np.arange(-100, 500)
    ax[0].plot(x_r80, gev.pdf(x_r80, shape, loc=loc, scale=scale), "k", lw=3, label='GEV (good fit)')
    ax[0].plot(x_r80, norm.pdf(x_r80, loc=loc, scale=scale), "r", lw=1, label='Normal (bad fit)', alpha=0.5)
    ax[0].axvline(return_level, color='b', label='return level', linestyle='--')
    ax[0].legend()

    # plot cdf
    x = np.linspace(-100, 500, 1000)
    y = gev.cdf(x, shape, loc=loc, scale=scale)
    ax[1].plot(x, y)
    ax[1].axhline(probability, color='b', linestyle='--')
    ax[1].axvline(return_level, color='b', linestyle='--')

    # plot return period vs return level
    return_periods = np.arange(2, 150, 5)
    return_levels, lower_return_levels, upper_return_levels = zip(*[evt.get_return_level(rp, with_confidence=True) for rp in return_periods])
    ax[2].plot(return_periods, return_levels)
    ax[2].fill_between(return_periods, lower_return_levels, upper_return_levels, alpha=0.2)
    ax[2].set_xlabel("Return Period")
    ax[2].set_ylabel("Return Level")
    ax[2].set_xscale("log")
    
    plt.savefig('imgs/gev.png')
    plt.show()



