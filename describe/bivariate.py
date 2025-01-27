import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

class BivariateAnalyzer:
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
        # Define correlation methods
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
        pval_df.columns = ["critical", "method", "significant"]

        # Add correlation and p-value columns for plotting
        pval_df["relation"] = pval_df["method"].apply(lambda x: self.results[x]["relation"])
        pval_df["p-value"] = pval_df["method"].apply(lambda x: self.results[x]["p-value"])
        # create unique coluln with method and p-value
        pval_df["method_pval"] = pval_df["method"] + " (p-value: " + pval_df["p-value"].astype(str) + ")"

        # plot the pval_df using a heatmap with index as critical, columns as method, and values as correlation and color as significant
        plt.figure(figsize=(10, 5))
        sns.heatmap(pval_df.pivot(index="critical", columns="method_pval", values="relation"),
                    # green if +1, red if -1, white if 0 from negative red to positive green
                    cmap="RdYlGn",
                    center=0,  # Center the color map at zero
                    annot=True,
                    fmt=".3f",
                    cbar=False,
                    linewidths=1,
                    linecolor="black",
                    vmin=-1,  # Ensure -1 is the minimum value
                    vmax=1,  # Ensure 1 is the maximum value
                    )
        plt.title(" Association between variables and significance levels")
        plt.show()

    def plot_scatter(self) -> None:
        sns.scatterplot(data=self.df, x=self.x, y=self.y)
        plt.show()

    def plot_kde(self) -> None:
        sns.kdeplot(data=self.df, x=self.x, y=self.y, fill=True)
        plt.show()

    def plot_pairplot(self) -> None:
        g = sns.pairplot(data=self.df, diag_kind="kde", markers=["o", "s"]) 
        g.map_lower(sns.kdeplot, levels=4, color=".2")
        plt.show()

    def plot_jointplot(self) -> None:
        sns.jointplot(data=self.df, x=self.x, y=self.y, kind="reg")
        plt.show()

    def plot_ecdf(self) -> None:
        sns.ecdfplot(data=self.df)
        plt.show()



if __name__ == "__main__":
    # create daframe with two variables negative correlated
    np.random.seed(42)
    n = 10000
    waiting = np.random.poisson(5, n)
    duration = 10 * waiting**2 + np.random.normal(0, 0.1, n)
    df = pd.DataFrame({"waiting": waiting, "duration": duration})
    
    params = {'n_neighbors': 15, 'num_permutations': 100, 'num_jobs': -1}
    bivariate = BivariateAnalyzer(df=df, x="waiting", y="duration", mutual_info=True, mi_params=params)
    results = bivariate.compute_association()
    bivariate.plot_association()
