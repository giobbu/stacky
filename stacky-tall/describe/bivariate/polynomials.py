import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class PolyAnalyzer:
    " Class for polynomial regression analysis of two variables. "
    def __init__(self, x: np.ndarray, y: np.ndarray, data: pd.DataFrame = None):
        " Initialize the PolyAnalyzer object. "
        self.x = x
        self.y = y
        self.df = data

    def __repr__(self):
        """
        Return a string representation of the PolyAnalyzer object.
        """
        return f"PolyAnalyzer(x={self.x}, y={self.y}, data={self.df})"

    def plot_polynomial_regr(self, order: int = 1, ci: int = 95, lowess_smoother_res: bool = False):
        """
        Plot a polynomial regression between two variables x and y.
        """
        coefs = np.polyfit(self.x, self.y, order).round(3)
        # two subplots
        # regression plot
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        x_name = self.df.columns[0] if self.df is not None else "x"
        y_name = self.df.columns[1] if self.df is not None else "y"
        sns.regplot(x=self.x, y=self.y, data=self.df, 
                    order=order, 
                    ci=ci, 
                    scatter_kws={"alpha": 0.1}, 
                    line_kws={"color": "green"},
                    ax=ax[0],
                    label=f"coefs: {coefs[:-1][::-1]}")
        ax[0].set_xlabel(x_name)
        ax[0].set_ylabel(y_name)
        ax[0].set_title(f"Polynomial regression plot between {x_name} and {y_name} with order: {order}")
        ax[0].grid(True)
        ax[0].legend()
        # plot residuals
        sns.residplot(x=self.x, y=self.y, 
        data=self.df,
            order=order, 
            scatter_kws={"alpha": 0.3}, 
            line_kws={"color": "red"}, 
            lowess=lowess_smoother_res, 
            ax=ax[1],
            label=f"coefs: {coefs[:-1][::-1]}")
        ax[1].set_xlabel(x_name)
        ax[1].set_ylabel("Residuals")
        ax[1].set_title(f"Residual plot between {x_name} and {y_name} with order: {order}")
        ax[1].grid(True)
        ax[1].legend()
        # save
        plt.savefig("imgs/polynomial_regr.png")
        plt.show()


if __name__=="__main__":
    # set seed for reproducibility
    np.random.seed(0)
    # drw random samples from a uniform distribution
    x = np.random.rand(1000)
    # linear relationship between x and y
    y = 2*x + np.random.normal(0, 0.1, 1000)
    # cubic relationship between x and y
    #y = x**3 + np.random.normal(0, 0.1, 1000)
    # create a dataframe xith columns names
    data = pd.DataFrame({"x": x, "y": y})
    # create a RegressionAnalyzer object
    ra = PolyAnalyzer(x, y, data)
    # plot the regression results
    ra.plot_polynomial_regr(ci=95, lowess_smoother_res=False, order=3)

