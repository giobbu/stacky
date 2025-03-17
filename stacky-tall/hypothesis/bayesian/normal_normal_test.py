import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from loguru import logger

class NormNormTest:
    """Class for Normal-Normal Bayesian A/B testing (no posterior for variance)"""
    def __init__(self, prior_params: list, threshold: float, means: list, stds: list, Ns: list):
        self.prior_params = prior_params
        self.threshold = threshold
        self.means = np.array(means)
        self.stds = np.array(stds)
        self.Ns = np.array(Ns)
        self.posteriors = self._compute_posteriors()
        self.xgrid_size = 1024
        self.eps = 0.1
        self.samples = 1000
        self.x = np.mgrid[self.means[1]-self.eps*self.stds[1]:self.means[1] + self.eps*self.stds[1]:self.xgrid_size*1j,
                            self.means[0]-self.eps*self.stds[0]:self.means[0] + self.eps*self.stds[0]:self.xgrid_size*1j]
        self.pdf_arr = self._compute_pdf()
    
    def _compute_posteriors(self):
        """Compute posterior distributions using Bayes' rule."""
        posteriors = []
        for i in range(len(prior_params)-1):
            inv_vars = self.prior_params[2][i] / np.power(self.prior_params[1][i], 2), self.Ns[i] / np.power(self.stds[i], 2)
            mu = np.average((self.prior_params[0][i], self.means[i]), weights=inv_vars)
            sd = 1 / np.sqrt(np.sum(inv_vars))
            posterior = norm(loc=mu, scale=sd)
            posteriors.append(posterior)
        return posteriors
    
    def _compute_pdf(self):
        """Compute the joint posterior distribution."""
        pdf_arr  = self.posteriors[0].pdf(self.x[1]) * self.posteriors[1].pdf(self.x[0])
        return pdf_arr/pdf_arr.sum()  # Normalize
    
    def plot_joint_posteriors(self):
        assert hasattr(self, 'risk_measure'), "Run the test first."
        plt.figure(figsize=(8, 6))
        plt.contourf(self.x[0], self.x[1], self.pdf_arr, levels=30, cmap="Blues")
        plt.colorbar(label="joint posterior density")
        plt.title(f"Joint Posterior Distribution with Risk Measure: {self.risk_measure}")
        plt.xlabel("Version B")
        plt.ylabel("Version A")
        # plot diagonal 45 degree line passing throught the mean of the two versions
        plt.plot([self.means[1], self.means[0]], [self.means[1], self.means[0]], "--k", alpha=0.5)
        # x-limit and y-limit
        plt.xlim(self.means[1]-self.eps*self.stds[1], self.means[1]+self.eps*self.stds[1])
        plt.ylim(self.means[0]-self.eps*self.stds[0], self.means[0]+self.eps*self.stds[0])
        # save the plot
        plt.savefig('imgs/joint_posterior_norm-norm.png')
        plt.show()

    def plot_posterior_marginals(self):
        assert hasattr(self, 'risk_measure'), "Run the test first."
        plt.figure(figsize=(8, 6))
        x_range_0 = np.linspace(self.means[0]-self.eps*self.stds[0], self.means[0]+self.eps*self.stds[0], self.samples)
        x_range_1 = np.linspace(self.means[1]-self.eps*self.stds[1], self.means[1]+self.eps*self.stds[1], self.samples)
        plt.plot(x_range_0, 
                self.posteriors[0].pdf(x_range_0), 
                label="Version A")
        plt.plot(x_range_1, 
                self.posteriors[1].pdf(x_range_1), 
                label="Version B")
        plt.xlabel("Metric of Interest")
        plt.ylabel("Density")
        plt.legend()
        plt.title(f"Posterior Marginal Distributions of Metric of Interest with Risk Measure: {self.risk_measure}")
        # save the plot
        plt.savefig('imgs/posterior_marginals_norm-norm.png')
        plt.show()

    def run_test(self):
        """Run the test and compute the risk measure."""
        prob_error = np.zeros_like(self.x[0])
        if (self.means[1] > self.means[0]):
            prob_error[np.where(self.x[1] > self.x[0])] = 1.0
        else:
            prob_error[np.where(self.x[0] > self.x[1])] = 1.0
        expected_error = np.maximum(np.abs(self.x[0] - self.x[1]), 0.0)
        self.risk_measure = round((expected_error * prob_error * self.pdf_arr).sum(), 4)
        # prob_larger = (prob_error * self.pdf_arr).sum()
        posterior_a = self.posteriors[0]
        posterior_b = self.posteriors[1]
        # samples from posteriors
        samples_a = posterior_a.rvs(self.samples)
        samples_b = posterior_b.rvs(self.samples)
        if self.risk_measure < self.threshold:
            if (self.means[1] > self.means[0]):
                logger.info(f"Version B is statistically significantly better than version A. with a risk measure of {self.risk_measure}")
                prob_larger = np.mean(samples_b > samples_a)
                logger.info("Probability that version B is better than version A: {:.2f}%".format(prob_larger * 100))
                rel_mean_lift = np.mean((samples_b - samples_a) / samples_a)
                logger.info("Relative mean lift of version B over version A: {:.2f}%".format(rel_mean_lift * 100))
            else:
                logger.info(f"Version A is statistically significantly better than version B. with a risk measure of {self.risk_measure}")
                prob_larger = np.mean(samples_a > samples_b)
                logger.info("Probability that version A is better than version B: {:.2f}%".format(prob_larger * 100))
                rel_mean_lift = np.mean((samples_a - samples_b) / samples_b)
                logger.info("Relative mean lift of version A over version B: {:.2f}%".format(rel_mean_lift * 100))
        else:
            logger.info("No version is statistically significantly better than the other.")
            logger.info('Continue testing.')
        # plot joint posterior
        self.plot_joint_posteriors()
        # plot posterior marginals
        self.plot_posterior_marginals()


# Example usage

# Data
m_a, s_a = 52.3, 14.1
m_b, s_b = 53.8, 13.7
N_a, N_b = 1000, 1004
means = [m_a, m_b]
stds = [s_a, s_b]
Ns = [N_a, N_b]
# Priors
m0_a, m0_b = 0, 0  # prior mean
s0_a, s0_b = 1, 1  # prior standard deviation
n0_a, n0_b = 0, 0  # prior sample size
prior_params = [(m0_a, m0_b), 
                (s0_a, s0_b), 
                (n0_a, n0_b)]
# set threshold
threshold = 0.01
test = NormNormTest(prior_params, threshold, means, stds, Ns)
test.run_test()