import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

class BinomialBetaTest:
    """Class for Binomial-Beta Bayesian A/B testing"""
    def __init__(self, prior_params, threshold, N, s):
        self.prior_params = prior_params
        self.threshold = threshold
        self.N = np.array(N)
        self.s = np.array(s)
        self.posteriors = self._compute_posteriors()
        self.xgrid_size = 1024
        self.x = np.mgrid[0:self.xgrid_size, 0:self.xgrid_size]/float(self.xgrid_size)
        self.pdf_arr = self._compute_pdf()
    
    def _compute_posteriors(self):
        """Compute posterior distributions using Bayes' rule."""
        posteriors = []
        for i in range(len(prior_params)):
            posterior = beta(prior_params[i][0] + s[i] - 1, prior_params[i][1] + N[i] - s[i] - 1) 
            posteriors.append(posterior)
        return posteriors
    
    def _compute_pdf(self):
        pdf_arr  = self.posteriors[0].pdf(self.x[1]) * self.posteriors[1].pdf(self.x[0])
        return pdf_arr / pdf_arr.sum()  # Normalize
    
    def plot_joint_posteriors(self):
        plt.figure(figsize=(8, 6))
        plt.contourf(self.x[0], self.x[1], self.pdf_arr, levels=50, cmap="Blues")
        plt.colorbar(label="joint posterior density")
        plt.plot([0, 1], [0, 1], "--k", alpha=0.5)
        plt.title("Joint Posterior Distribution")
        plt.xlabel("Version B")
        plt.ylabel("Version A")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        # save the plot
        plt.savefig('imgs/joint_posterior.png')
        plt.show()

    def plot_posterior_marginals(self):
        plt.figure(figsize=(8, 6))
        samples = 1000
        plt.plot(np.linspace(0, 1, samples), self.posteriors[0].pdf(np.linspace(0, 1, samples)), label="Version A")
        plt.plot(np.linspace(0, 1, samples), self.posteriors[1].pdf(np.linspace(0, 1, samples)), label="Version B")
        plt.xlabel("Metric of Interest")
        plt.ylabel("Density")
        plt.legend()
        plt.title("Posterior Marginal Distributions of Metric of Interest")
        # save the plot
        plt.savefig('imgs/posterior_marginals.png')
        plt.show()

    def run_test(self):
        prob_error = np.zeros_like(self.x[0])
        if (self.s[1] / float(self.N[1])) > (self.s[0] / float(self.N[0])):
            prob_error[np.where(self.x[1] > self.x[0])] = 1.0
        else:
            prob_error[np.where(self.x[0] > self.x[1])] = 1.0
        
        expected_error = np.maximum(np.abs(self.x[0] - self.x[1]), 0.0)
        expected_err_scalar = (expected_error * prob_error * self.pdf_arr).sum()
        prob_larger = (prob_error * self.pdf_arr).sum()
        
        if expected_err_scalar < self.threshold:
            if (s[1]/float(N[1])) > (s[0]/float(N[0])):
                lift_B = (s[1]/float(N[1]) - s[0]/float(N[0]))/ (s[0]/float(N[0])) - 1
                print("Version B is statistically significantly better than version A.")
                print("Expected lift: {:.2f}%".format(lift_B))
                print("Probability that version B is better than version A: {:.2f}%".format(prob_larger * 100))
            else:
                lift_A = (s[0]/float(N[0]) - s[1]/float(N[1]))/ (s[1]/float(N[1])) - 1
                print("Version A is statistically significantly better than version B.")
                print("Expected lift: {:.2f}%".format(lift_A))
                print("Probability that version A is better than version B: {:.2f}%".format(prob_larger * 100))
        else:
            print("No version is statistically significantly better than the other.")
            print('Continue testing.')


# Example usage
prior_params = [(1, 1), (1, 1)]
threshold = 0.001
N = [200, 204]
s = [16, 36]

test = BinomialBetaTest(prior_params, threshold, N, s)
test.run_test()
test.plot_joint_posteriors()
test.plot_posterior_marginals()