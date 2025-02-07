import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class GaussianCopula:
    " Class for generating Gaussian copula distribution. "
    def __init__(self, marginal_x, marginal_y, means, cov_matrix):
        " Initialize the GaussianCopula object. "
        self.marginal_x = marginal_x
        self.marginal_y = marginal_y
        self.means = means
        self.cov_matrix = cov_matrix

    def _multivariate_gaussian(self):
        " Generate multivariate Gaussian distribution. "
        mv_norm = stats.multivariate_normal(self.means, self.cov_matrix)
        mv_samples = mv_norm.rvs(10000)
        return mv_samples

    def _copula_transform_uniform(self):
        " Transform multivariate Gaussian to uniform distribution. "
        mv_samples = self._multivariate_gaussian()
        norm = stats.norm()
        uniforms = norm.cdf(mv_samples)
        return uniforms
    
    def transform(self):
        " Transform uniform distribution to marginals. "
        uniforms = self._copula_transform_uniform()
        x_transformed = self.marginal_x.ppf(uniforms[:, 0])
        y_transformed = self.marginal_y.ppf(uniforms[:, 1])
        return x_transformed, y_transformed
    
    def plot_copula(self):
        " Plot the copula distribution. "
        x_trans, y_trans = self.transform()
        data = pd.DataFrame({'x': x_trans, 'y': y_trans})
        g = sns.jointplot(data=data, x='x', y='y', kind='kde', color='blue', fill=True)
        g.set_axis_labels('X', 'Y')
        g.fig.suptitle('Gaussian Copula')
        plt.show()


    def plot_comparison(self):
        " Plot comparison between marginals and copula distribution. "
        # marginals
        x = self.marginal_x.rvs(10000)
        y = self.marginal_y.rvs(10000)
        marginal_data = pd.DataFrame({'x': x, 'y': y})
        # copula
        x_trans, y_trans = self.transform()
        copula_data = pd.DataFrame({'x_trans':x_trans, 'y_trans': y_trans})
        # Create figure for side-by-side plots
        # Marginal Distribution
        g1 = sns.jointplot(data=marginal_data, x='x', y='y', kind='kde', color='red', fill=True)
        g1.set_axis_labels('X', 'Y')
        g1.fig.suptitle('Marginals', fontsize=14)
        g1.savefig('imgs/marginals.png')
        # Copula Distribution
        g2 = sns.jointplot(data=copula_data, x='x_trans', y='y_trans', kind='kde', color='blue', fill=True)
        g2.set_axis_labels('X (Transformed)', 'Y (Transformed)')
        g2.fig.suptitle('Copula', fontsize=14)
        # save plot
        g2.savefig('imgs/copula.png')
        plt.show()
        

if __name__=="__main__":
    # set seed for reproducibility
    np.random.seed(0)
    # create marginal distributions
    marginal_x = stats.gumbel_l()  # Gumbel left distribution
    marginal_y = stats.beta(a=10, b=2)  # Beta distribution
    # create means and covariance matrix
    means = [0, 0]
    cov_matrix = [[1, 0.5], [0.5, 1]]

    # create GaussianCopula object
    copula = GaussianCopula(marginal_x, marginal_y, means, cov_matrix)

    # plot copula
    #copula.plot_copula()

    # plot comparison
    copula.plot_comparison()
