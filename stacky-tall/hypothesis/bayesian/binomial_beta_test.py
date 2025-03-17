import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from loguru import logger

class BinomialBetaTest:
    """Class for Binomial-Beta Bayesian A/B testing"""
    def __init__(self, prior_params: list, threshold: float, N: list, s: list):
        self.prior_params = prior_params
        self.threshold = threshold
        self.N = np.array(N)
        self.s = np.array(s)
        self.posteriors = self._compute_posteriors()
        self.xgrid_size = 1024
        self.x = np.mgrid[0:self.xgrid_size, 0:self.xgrid_size]/float(self.xgrid_size)
        self.pdf_arr = self._compute_pdf()
        self.samples = 1000
    
    def _compute_posteriors(self):
        """Compute posterior distributions using Bayes' rule."""
        posteriors = []
        for i in range(len(prior_params)):
            posterior = beta(prior_params[i][0] + s[i] - 1, prior_params[i][1] + N[i] - s[i] - 1) 
            posteriors.append(posterior)
        return posteriors
    
    def _compute_pdf(self):
        """Compute the joint posterior distribution."""
        pdf_arr  = self.posteriors[0].pdf(self.x[1]) * self.posteriors[1].pdf(self.x[0])
        return pdf_arr / pdf_arr.sum()  # Normalize
    
    def plot_joint_posteriors(self):
        assert hasattr(self, 'risk'), "Run the test first."
        plt.figure(figsize=(8, 6))
        plt.contourf(self.x[0], self.x[1], self.pdf_arr, levels=50, cmap="Blues")
        plt.colorbar(label="joint posterior density")
        plt.plot([0, 1], [0, 1], "--k", alpha=0.5)
        plt.title(f"Joint Posterior Distribution with Risk Measure: {self.risk}")
        plt.xlabel("Version B")
        plt.ylabel("Version A")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        # save the plot
        plt.savefig('imgs/joint_posterior_binom-beta.png')
        plt.show()

    def plot_posterior_marginals(self):
        assert hasattr(self, 'risk'), "Run the test first."
        plt.figure(figsize=(8, 6))
        plt.plot(np.linspace(0, 1, self.samples), self.posteriors[0].pdf(np.linspace(0, 1, self.samples)), label="Version A")
        plt.plot(np.linspace(0, 1, self.samples), self.posteriors[1].pdf(np.linspace(0, 1, self.samples)), label="Version B")
        plt.xlabel("Metric of Interest")
        plt.ylabel("Density")
        plt.legend()
        plt.title(f"Posterior Marginal Distributions of Metric of Interest with Risk Measure: {self.risk}")
        # save the plot
        plt.savefig('imgs/posterior_marginals_binom-beta.png')
        plt.show()

    def run_test(self):
        """Run the test and compute the risk measure."""
        prob_error = np.zeros_like(self.x[0])
        if (self.s[1] / float(self.N[1])) > (self.s[0] / float(self.N[0])):
            prob_error[np.where(self.x[1] > self.x[0])] = 1.0
        else:
            prob_error[np.where(self.x[0] > self.x[1])] = 1.0
        expected_error = np.maximum(np.abs(self.x[0] - self.x[1]), 0.0)
        self.risk = round((expected_error * prob_error * self.pdf_arr).sum(), 4)
        # samples from the posteriors
        samples_a = self.posteriors[0].rvs(self.samples)
        samples_b = self.posteriors[1].rvs(self.samples)
        if self.risk < self.threshold:
            if (s[1]/float(N[1])) > (s[0]/float(N[0])):
                logger.info("Version B is statistically significantly better than version A.")
                # compute the probability that version B is better than version A
                prob_larger = (samples_b > samples_a).mean()
                logger.info("Probability that version B is better than version A: {:.2f}%".format(prob_larger * 100))
                # compute relative lift
                relative_mean_lift = np.mean((samples_b - samples_a) / samples_a)
                logger.info("Relative mean lift of version B over version A: {:.2f}%".format(relative_mean_lift * 100))
            else:
                logger.info("Version A is statistically significantly better than version B.")
                # compute the probability that version A is better than version B
                prob_larger = (samples_a > samples_b).mean()
                logger.info("Probability that version A is better than version B: {:.2f}%".format(prob_larger * 100))
                # compute relative lift
                relative_mean_lift = np.mean((samples_a - samples_b) / samples_b)
                logger.info("Relative mean lift of version A over version B: {:.2f}%".format(relative_mean_lift * 100))
        else:
            logger.info("No version is statistically significantly better than the other.")
            logger.info('Continue testing.')
        # plot joint posterior
        self.plot_joint_posteriors()
        # plot posterior marginals
        self.plot_posterior_marginals()


# Example usage
prior_params = [(1, 1), (1, 1)]
threshold = 0.001
N = [200, 204]
s = [16, 36]

test = BinomialBetaTest(prior_params, threshold, N, s)
test.run_test()
