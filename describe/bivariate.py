import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

class BivariateAnalyzer:
    " Class for analysis of two variables. "

    def __init__(self, df: pd.DataFrame, x: str, y: str) -> None:
        self.df = df
        self.x = x
        self.y = y

    def plot_corr_scipy(self) -> dict:
        """
        Calculate and plot the Pearson, Spearman, and Kendall correlation coefficients and their p-values.

        Returns:
            dict: A dictionary containing the correlation coefficients and p-values for each method.
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd

        # Define correlation methods
        methods = {
            'Pearson': stats.pearsonr,
            'Spearman': stats.spearmanr,
            'Kendall': stats.kendalltau
        }

        # Compute correlations and p-values for each method
        results = {}
        for method, func in methods.items():
            corr, pval = func(self.df[self.x], self.df[self.y])
            results[method] = {
                'correlation': round(corr, 3),
                'p-value': round(pval, 4)
            }

        # Define critical thresholds
        critical_values = [0.001, 0.01, 0.05]

        # Prepare a DataFrame for critical value significance analysis
        pval_df = pd.DataFrame(index=[str(threshold) for threshold in critical_values])

        for method, result in results.items():
            pval_df[method] = [
                1 if result['p-value'] <= threshold else 0 for threshold in critical_values
            ]

        # Reset index and reshape for plotting
        pval_df = pval_df.reset_index().melt(id_vars="index", var_name="method", value_name="significant")
        pval_df.columns = ["critical", "method", "significant"]

        # Add correlation and p-value columns for plotting
        pval_df["correlation"] = pval_df["method"].apply(lambda x: results[x]["correlation"])
        pval_df["p-value"] = pval_df["method"].apply(lambda x: results[x]["p-value"])
        # create unique coluln with method and p-value
        pval_df["method_pval"] = pval_df["method"] + " (p-value: " + pval_df["p-value"].astype(str) + ")"

        # plot the pval_df using a heatmap with index as critical, columns as method, and values as correlation and color as significant
        plt.figure(figsize=(10, 5))
        sns.heatmap(pval_df.pivot(index="critical", columns="method_pval", values="correlation"),
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
        
        plt.title("SCIPY: Correlation (without p-value)")
        plt.show()



        return results


        
    
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
    np.random.seed(0)
    n = 1000
    waiting = np.random.poisson(5, n)
    duration = 10 + waiting + np.random.normal(0, 1, n)
    df = pd.DataFrame({"waiting": waiting, "duration": duration})
    bivariate = BivariateAnalyzer(df=df, x="waiting", y="duration")
    # bivariate.plot_scatter()
    # bivariate.plot_kde()
    # bivariate.plot_pairplot()
    # bivariate.plot_jointplot()
    # bivariate.plot_ecdf()
    # bivariate.plot_corr_pandas()
    bivariate.plot_corr_scipy()
