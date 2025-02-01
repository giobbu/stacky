import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

class VanillaBivariateAnalyzer:
    " Class for analysis of two variables. "
    def __init__(self, df: pd.DataFrame, x: str, y: str, mutual_info: bool = False, mi_params: dict = None) -> None:
        " Initialize the BivariateAnalyzer object. "
        self.df = df
        self.x = x
        self.y = y
        self.mutual_info_enabled = mutual_info
        self.mi_params = mi_params or {"n_neighbors":3, 'num_permutations':100, 'n_jobs':-1} 

    def __repr__(self):
        """
        Return a string representation of the BivariateAnalyzer object.
        """
        return f"BivariateAnalyzer(x={self.x}, y={self.y}, mutual_info_enabled={self.mutual_info_enabled}, mi_params={self.mi_params})"

    def _compute_mutual_info(self) -> tuple[float, float]:
        """
        Compute mutual information between two variables and a p-value.

        :return: A tuple (mutual information score, p-value).
        """
        mi_base = self._compute_mi_score(self.df)
        abs_mi_base = np.abs(mi_base)
        # Permutation testing for p-value calculation
        permuted_scores = []
        for _ in range(self.mi_params["num_permutations"]):
            df_permuted = self.df.copy()
            df_permuted[self.x] = np.random.permutation(df_permuted[self.x].values)
            mi_permuted = self._compute_mi_score(df_permuted)
            permuted_scores.append(np.abs(mi_permuted))
        # Calculate p-value
        sum_  = (abs_mi_base >= np.array(permuted_scores)).sum()
        freq_greater = sum_ / self.mi_params["num_permutations"]
        p_value = 1 - freq_greater
        return float(round(mi_base, 3)), float(round(p_value, 4))

    def _compute_mi_score(self, df: pd.DataFrame) -> float:
        """
        Compute mutual information score between two variables.

        :param df: The DataFrame containing the variables.
        :return: The mutual information score.
        """
        return mutual_info_regression(
            X=df[[self.x]], 
            y=df[self.y], 
            n_neighbors=self.mi_params["n_neighbors"], 
            random_state=0, 
            #n_jobs=-1#self.mi_params["n_jobs"]
        )[0]


    def compute_association(self) -> dict:
        " Compute the strength coefficients and p-values for the two variables. "
        methods = {
            'Pearson': stats.pearsonr,
            'Spearman': stats.spearmanr,
            'Kendall': stats.kendalltau
        }

        # Compute correlations and p-values for each method
        self.results = {}
        for method, func in methods.items():
            corr, pval = func(self.df[self.x], self.df[self.y])
            self.results[method] = {
                'relation': float(round(corr, 3)),
                'p-value': float(round(pval, 4))
            }
        # Add mutual information results if enabled
        if self.mutual_info_enabled:
            mi_score, p_value = self._compute_mutual_info()
            self.results['Mut Info'] = {
                'relation': mi_score,
                'p-value': p_value
            }
        return self.results

    def plot_association(self) -> dict:
        """
        Calculate and plot the Pearson, Spearman, and Kendall correlation coefficients and their p-values.
        Returns:
            dict: A dictionary containing the correlation coefficients and p-values for each method.
        """
        # Define critical thresholds
        critical_values = [0.001, 0.01, 0.05]
        # Prepare a DataFrame for critical value significance analysis
        pval_df = pd.DataFrame(index=[str(threshold) for threshold in critical_values])
        for method, result in self.results.items():
            pval_df[method] = [
                1 if result['p-value'] <= threshold else 0 for threshold in critical_values
            ]
        # Reset index and reshape for plotting
        pval_df = pval_df.reset_index().melt(id_vars="index", var_name="method", value_name="significant")
        pval_df.columns = ["critical levels", "method", "significant"]
        # Add correlation and p-value columns for plotting
        pval_df["relation strength"] = pval_df["method"].apply(lambda x: self.results[x]["relation"])
        pval_df["p-value"] = pval_df["method"].apply(lambda x: self.results[x]["p-value"])
        # create unique coluln with method and p-value
        pval_df["method (p-value)"] = pval_df["method"] + " (p-value: " + pval_df["p-value"].astype(str) + ")"
        # plot the pval_df using a heatmap with index as critical, columns as method, and values as correlation and color as significant
        plt.figure(figsize=(10, 5))
        sns.heatmap(pval_df.pivot(index="critical levels", columns="method (p-value)", values="relation strength"),
                    # green if +1, red if -1, white if 0 from negative red to positive green
                    cmap="RdYlGn",
                    center=0,  # Center the color map at zero
                    annot=True,
                    fmt=".3f",
                    cbar=False,
                    linewidths=3,
                    linecolor="white",
                    vmin=-1,  # Ensure -1 is the minimum value
                    vmax=1,  # Ensure 1 is the maximum value
                    )
        plt.title(" Association between two variables with significance levels")
        plt.show()

    def plot_eda(self) -> None:
        " Plot a jointplot of the two variables. "

        # plot ecdf of x
        sns.ecdfplot(data=self.df,)
        plt.title("Empirical Cumulative Distribution Function")
        plt.show()
        
        # plot violinplot
        sns.violinplot(data=self.df, alpha=0.1)
        # add stripplot
        sns.stripplot(data=self.df, alpha=0.5)
        plt.title("Violinplot of the two variables")
        plt.show()


        # plot jointplot
        g = sns.jointplot(data=self.df, x=self.x, y=self.y, alpha=0.5)
        # add plot_joint
        g.plot_joint(sns.kdeplot, zorder=0, levels=6, fill = True, cmap="viridis")
        plt.show()



if __name__ == "__main__":
    # create daframe with two variables negative correlated
    np.random.seed(42)
    n = 1000
    waiting = np.random.uniform(0, 100, n)
    duration = np.cos(100 * waiting) + np.random.normal(5, 10, n)
    df = pd.DataFrame({"waiting": waiting, "duration": duration})
    # create BivariateAnalyzer object
    params = {'n_neighbors': 15, 'num_permutations': 100, 'num_jobs': -1}
    bivariate = VanillaBivariateAnalyzer(df=df, x="waiting", y="duration", mutual_info=True, mi_params=params)
    # plot jointplot
    bivariate.plot_eda()
    # compute and plot association
    results = bivariate.compute_association()  # calculate association
    bivariate.plot_association()  # plot association
