import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class BivariateAnalyzer:
    " Class for analysis of two variables. "
    def __init__(self, df: pd.DataFrame, x: str, y: str) -> None:
        self.df = df
        self.x = x
        self.y = y

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

    def plot_corr(self) -> None:
        # plot pearson, spearman and kendall correlation matrix in one figure
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        corr = pd.DataFrame(self.df.corr().iloc[0, 1].reshape(1, 1), index=[self.y], columns=[self.x])
        sns.heatmap(corr, annot=True, ax=ax[0], cbar=False, cmap="warm")
        ax[0].set_title("Pearson Correlation")
        corr = pd.DataFrame(self.df.corr(method="spearman").iloc[0, 1].reshape(1, 1), index=[self.y], columns=[self.x])
        sns.heatmap(corr, annot=True, ax=ax[1], cbar=False, cmap="warm")
        ax[1].set_title("Spearman Correlation")
        corr = pd.DataFrame(self.df.corr(method="kendall").iloc[0, 1].reshape(1, 1), index=[self.y], columns=[self.x])
        sns.heatmap(corr, annot=True, ax=ax[2], cbar=False, cmap= "warm")
        ax[2].set_title("Kendall Correlation")
        plt.show()


if __name__ == "__main__":
    df = sns.load_dataset("geyser")[["waiting", "duration"]]
    bivariate = BivariateAnalyzer(df=df, x="waiting", y="duration")
    bivariate.plot_kde()
    bivariate.plot_pairplot()
    bivariate.plot_jointplot()
    bivariate.plot_ecdf()
    bivariate.plot_corr()
