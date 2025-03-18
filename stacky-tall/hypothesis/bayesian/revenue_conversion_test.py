import numpy as np

class RevenueABTest:
    """Class for Revenue A/B testing"""
    def __init__(self, priors_beta: list, priors_gamma: list, threshold: float, N: list, n: list):
        self.priors_beta = priors_beta
        self.priors_gamma = priors_gamma
        self.N = np.array(N)  # total number
        self.n = n  # number of conversions
        self.threshold = threshold

    def binomial_beta_posteriors(self):
        """conversion probability 𝜆: 𝑋 ∼ 𝐵𝑒𝑟(𝜆)"""
        pass

    def exponential_gamma_posteriors(self):
        """rate parameter 𝜃: 𝑌 ∼ 𝐸𝑥𝑝(𝜃)"""
        pass

    def revenue_posteriors(self):
        """average revenue 𝜆/𝜃"""
        pass

    def run_test(self):
        pass