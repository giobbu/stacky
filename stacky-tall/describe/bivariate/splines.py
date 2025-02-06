import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import make_lsq_spline, generate_knots
from sklearn.model_selection import train_test_split


class SplinesAnalyzer:
    " Class for splines regression analysis of two variables. "
    def __init__(self, x: np.ndarray, y: np.ndarray, data: pd.DataFrame = None):
        " Initialize the SplineAnalyzer object. "
        self.x = x
        self.y = y
        self.df = data

    def __repr__(self):
        """
        Return a string representation of the SplineAnalyzer object.
        """
        return f"SplineAnalyzer(x={self.x}, y={self.y}, data={self.df})"
    
    def _compute_splines_coefs(self, s: float, t: np.ndarray, size_test: float = 0.1):
        """
        Compute the coefficients of the spline function.
        """
        # Split data into training and test sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=size_test, random_state=42)
        best_t = None
        best_g = None
        best_rmse = float('inf')  # Initialize with a large value
        # ordered 1D sequences x and y
        x_train_sorted = np.sort(x_train)
        y_train_sorted = y_train[np.argsort(x_train)]
        for t in generate_knots(x_train_sorted, y_train_sorted, s=s):  
            g = make_lsq_spline(x_train_sorted, y_train_sorted, t=t)  # Fit spline using training set
            y_pred = g(x_test)  # Predict on test set
            rmse = np.sqrt(((y_test - y_pred) ** 2).mean())  # Compute RMSE on test set
            if rmse < best_rmse:  # Choose the best fit based on smallest test RMSE
                best_t, best_g, best_rmse = t, g, rmse
        return best_t, best_g, best_rmse, x_train, x_test, y_train, y_test
    
    def plot_splines_regr(self, s: float = 5, t: np.ndarray = None, size_test: float = 0.1):
        """
        Plot a spline regression between two variables x and y.
        """
        _, best_g, best_rmse, x_train, x_test, y_train, y_test = self._compute_splines_coefs(s, t, size_test)
        # Plotting
        plt.figure(figsize=(10, 6))
        # Plot training points in orange
        sns.scatterplot(x=x_train, y=y_train, color='green', label='Train Points', s=50, edgecolor='black')
        # Plot test points in green
        sns.scatterplot(x=x_test, y=y_test, color='orange', label='Test Points', s=50, edgecolor='black')
        # Plot the best spline in blue
        x_line = np.linspace(np.min(x), np.max(x), 1000)
        y_line = best_g(x_line)
        plt.plot(x_line, y_line, color='blue', label='Best Spline (RMSE={:.2f})'.format(best_rmse), lw=2)
        # Add labels and title
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Train/Test Points and Best Spline')
        # Add legend
        plt.legend()
        # save
        plt.savefig('imgs/splines_regr.png')
        # Display the plot
        plt.show()


if __name__=="__main__":
    # Set random seed for reproducibility
    np.random.seed(2)
    x = np.random.normal(0, 1, 1000)
    y = np.cos(x) + np.random.normal(0, 0.1, 1000)
    s = 5  # Smoothing factor
    # Create a DataFrame
    data = pd.DataFrame({"x": x, "y": y})
    # Create a SplineAnalyzer object
    sa = SplinesAnalyzer(x, y, data)
    # Plot the spline regression
    sa.plot_splines_regr(s=s, t=None, size_test=0.1)
    