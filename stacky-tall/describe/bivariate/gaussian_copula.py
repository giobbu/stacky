import numpy as np
import scipy.stats as stats
from scipy.stats import gaussian_kde
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF


class GaussianCopulaDemo:
    " Class for generating Gaussian copula distribution. "

    def __init__(self, marginal_x, marginal_y, means, cov_matrix, num_samples=100000):
        " Initialize the GaussianCopula object. "
        self.marginal_x = marginal_x
        self.marginal_y = marginal_y
        self.means = means
        self.cov_matrix = cov_matrix
        self.num_samples = num_samples

    def _generate_multivariate_gaussian(self):
        " Generate multivariate Gaussian distribution. "
        mv_samples = stats.multivariate_normal(self.means, self.cov_matrix).rvs(self.num_samples)  # generate multivariate Gaussian samples
        return mv_samples

    def _multivariate_to_uniforms(self):
        " Transform multivariate Gaussian to uniform distribution. "
        samples = self._generate_multivariate_gaussian()
        uniforms = stats.norm().cdf(samples)  # transform to uniform distribution
        return uniforms

    def copula_to_marginals(self):
        " Transform uniform distribution to marginals. "
        uniforms = self._multivariate_to_uniforms()
        x_transformed = self.marginal_x.ppf(uniforms[:, 0])
        y_transformed = self.marginal_y.ppf(uniforms[:, 1])
        return x_transformed, y_transformed

    def get_rank_correlation(self):
        " Get rank correlation between marginals. "
        rank_corr_kendal = 2/np.pi * np.arcsin(self.cov_matrix[0][1])
        rank_corr_spearman = 6/np.pi * np.arcsin(self.cov_matrix[0][1]/2)
        return rank_corr_kendal, rank_corr_spearman
    
    def plot_comparison(self):
        " Plot comparison between marginals and copula distribution. "
        
        # marginals & rank correlation
        x = self.marginal_x.rvs(self.num_samples)
        y = self.marginal_y.rvs(self.num_samples)
        
        rank_corr_kendal = stats.kendalltau(x, y).statistic
        rank_corr_spearman = stats.spearmanr(x, y).statistic
        marginal_data = pd.DataFrame({'x': x, 'y': y})
        
        # copula data & copula rank correlation
        x_trans, y_trans = self.copula_to_marginals()
        copula_data = pd.DataFrame({'x_trans':x_trans, 'y_trans': y_trans})
        copula_rank_corr_kendal, copula_rank_corr_spearman = self.get_rank_correlation()

        # Marginal Distribution
        g1 = sns.jointplot(data=marginal_data, x='x', y='y', kind='kde', color='red', fill=True)
        g1.set_axis_labels('X', 'Y')
        g1.fig.suptitle(F'Marginal Distributions with Kendall Rank Correlation: {rank_corr_kendal:.2f} and Spearman Rank Correlation: {rank_corr_spearman:.2f}', fontsize=14)
        g1.savefig('imgs/marginals.png')

        # Copula Distribution
        g2 = sns.jointplot(data=copula_data, x='x_trans', y='y_trans', kind='kde', color='blue', fill=True)
        g2.set_axis_labels('X (Transformed)', 'Y (Transformed)')
        g2.fig.suptitle(f'Copula with Kendall Rank Correlation: {copula_rank_corr_kendal:.2f} and Spearman Rank Correlation: {copula_rank_corr_spearman:.2f}', fontsize=14)
        g2.savefig('imgs/copula.png')

    def _estimate_copula_pdf(self):
        " Estimate copula PDF using Gaussian copula. "
        u_range = np.linspace(0.001, .999, 100)
        v_range = np.linspace(0.001, .999, 100)
        # Compute inverse normal CDF (quantile function)
        inv_u_range = stats.norm.ppf(u_range)
        inv_v_range = stats.norm.ppf(v_range)
        rho = self.cov_matrix[0][1]  # Extract correlation coefficient
        # Create a grid of values
        inv_u_grid, inv_v_grid = np.meshgrid(inv_u_range, inv_v_range)
        u_grid, v_grid = np.meshgrid(u_range, v_range)
        exp_ker = (rho * (-2 * inv_u_grid * inv_v_grid + inv_u_grid ** 2 * rho + inv_v_grid ** 2 * rho)/ (2 * (rho ** 2 - 1)))
        pdf_grid = np.exp(exp_ker) / np.sqrt(1 - rho ** 2)
        return pdf_grid, u_grid, v_grid

    def plot_Copula_PDF(self):
        " Plot 3D copula distribution. "
        copula_density, u_grid, v_grid = self._estimate_copula_pdf()
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(u_grid, v_grid, copula_density, cmap='viridis')
        ax.set_xlabel('u')
        ax.set_ylabel('v')
        ax.set_zlabel('PDF')
        ax.set_title('Gaussian Copula PDF')
        # set limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.savefig('imgs/copula_pdf.png')
        plt.show()
    
    def _estimate_copula(self, u, v, U, V):
        " Estimate empirical copula. "
        return np.mean((U <= u) & (V <= v))
    
    def plot_Copula_ECDF(self):
        " Plot empirical copula distribution. "
        U, V = self._multivariate_to_uniforms().T
        u_vals = np.linspace(0.001, .999, 100)
        v_vals = np.linspace(0.001, .999, 100)
        U_grid, V_grid = np.meshgrid(u_vals, v_vals)
        C_empirical = np.array([[self._estimate_copula(u, v, U, V) for u in u_vals] for v in v_vals])
        # plot empirical copula
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(U_grid, V_grid, C_empirical, cmap='viridis')
        ax.set_xlabel('U (Transformed X)')
        ax.set_ylabel('V (Transformed Y)')
        ax.set_title('Gaussian Copula CDF')
        plt.savefig('imgs/copula_ecdf.png')
        plt.show()

    def plot_Copula_Hex(self):
        samples = self._generate_multivariate_gaussian()
        uniforms = stats.norm().cdf(samples)  # transform to uniform distribution
        # plot jointplot
        sns.jointplot(data=pd.DataFrame(uniforms, columns=['u', 'v']), x='u', y='v', kind='hex', color='green')
        plt.xlabel('u')
        plt.ylabel('v')
        plt.savefig('imgs/copula_hex.png')
        plt.show()

    def plot_Copula_Kde(self):
        samples = self._generate_multivariate_gaussian()
        uniforms = stats.norm().cdf(samples)  # transform to uniform distribution
        # plot jointplot
        sns.jointplot(data=pd.DataFrame(uniforms, columns=['u', 'v']), x='u', y='v', kind='kde', color='green', fill=True)
        plt.xlabel('u')
        plt.ylabel('v')
        plt.savefig('imgs/copula_kde.png')
        plt.show()

if __name__=="__main__":

    # set seed for reproducibility
    np.random.seed(2)

    # create marginal distributions Beta and Gumbel
    marginal_x = stats.beta(a=2, b=5)
    marginal_y = stats.gumbel_r(loc=0, scale=1)

    # create means and covariance matrix
    means = [0, 0]
    cov_matrix = [[1., 0.3], [0.3, 1.]]

    # create GaussianCopula object
    copula = GaussianCopulaDemo(marginal_x, marginal_y, means, cov_matrix)

    copula.plot_Copula_PDF()

    # plot copula ECDF
    copula.plot_Copula_ECDF()

    # # plot copula
    # copula.plot_comparison()

    # # plot copula PDF
    # copula.plot_Copula_PDF()

    # # plot copula hex
    # copula.plot_Copula_Hex()

