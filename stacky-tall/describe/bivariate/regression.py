import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class RegressionAnalyzer:
    " Class for regression analysis of two variables. "
    def __init__(self, x: np.ndarray, y: np.ndarray, data: pd.DataFrame = None):
        " Initialize the RegressionBivariateAnalyzer object. "
        self.x = x
        self.y = y
        self.df = data

    def __repr__(self):
        """
        Return a string representation of the RegressionBivariateAnalyzer object.
        """
        return f"RegressionBivariateAnalyzer(x={self.x}, y={self.y})"

    def plot_regression(self, order: int = 1, robust: bool = False, ci: int = 95, lowess_regr: bool = False, lowess_smoother_res: bool = False):
        """
        Plot a regression plot between two variables.
        """
        # two subplots
        # regression plot
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        x_name = self.df.columns[0] if self.df is not None else "x"
        y_name = self.df.columns[1] if self.df is not None else "y"
        sns.regplot(x=self.x, y=self.y, data=self.df, 
                    order=order, 
                    robust=robust, 
                    ci=ci, 
                    scatter_kws={"alpha": 0.3}, 
                    line_kws={"color": "green"}, 
                    lowess=lowess_regr, ax=ax[0],
                    label=f"Order: {order}")
        ax[0].set_xlabel(x_name)
        ax[0].set_ylabel(y_name)
        ax[0].set_title(f"Regression plot between {x_name} and {y_name}")
        ax[0].grid(True)
        # plot residuals
        sns.residplot(x=self.x, y=self.y, 
        data=self.df,
            order=order, 
            scatter_kws={"alpha": 0.3}, 
            line_kws={"color": "red"}, 
            lowess=lowess_smoother_res, 
            ax=ax[1],
            label=f"Order: {order}")
        ax[1].set_xlabel(x_name)
        ax[1].set_ylabel("Residuals")
        ax[1].set_title(f"Residual plot between {x_name} and {y_name}")
        ax[1].grid(True)
        plt.show()


if __name__=="__main__":
    # set seed for reproducibility
    np.random.seed(0)
    # drw random samples from a uniform distribution
    x = np.random.rand(1000)
    # linear relationship between x and y
    y = 2*x + np.random.normal(0, 0.1, 1000)
    # cubic relationship between x and y
    y = x**3 + np.random.normal(0, 0.1, 1000)
    # create a dataframe xith columns names
    data = pd.DataFrame({"x": x, "y": y})
    # create a RegressionAnalyzer object
    ra = RegressionAnalyzer(x, y, data)
    # plot the regression results
    ra.plot_regression(ci=95, lowess_regr=True, lowess_smoother_res=False)
