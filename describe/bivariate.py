import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class BivariateAnalyzer:
    " Class for analysis of two variables. "
    def __init__(self, df: pd.DataFrame, x: str, y: str) -> None:
        self.df = df
        self.x = x
        self.y = y

    def plot_kde(self) -> None:
        sns.kdeplot(data=self.df, x=self.x, y=self.y, hue="kind", fill=True)
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
    df = sns.load_dataset("geyser")
    bivariate = BivariateAnalyzer(df=df, x="waiting", y="duration")
    bivariate.plot_kde()
    bivariate.plot_pairplot()
    bivariate.plot_jointplot()
    bivariate.plot_ecdf()
