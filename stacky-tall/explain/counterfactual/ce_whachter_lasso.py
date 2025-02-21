import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from scipy.optimize import minimize
from loguru import logger

class WachterCFLasso:
    " Counterfactual optimization for Lasso regression. "
    def __init__(self, cv=2, max_iter=10000, params_grid=None):
        self.cv = cv
        self.max_iter = max_iter
        self.params_grid = params_grid

    def fit(self, X, y):
        """
        Fit a Lasso regression model to the data and return the model's coefficients (w) and intercept (b).
        """
        # Fit the Lasso model with cross-validation
        lasso = Lasso(max_iter=self.max_iter)
        grid_search = GridSearchCV(lasso, self.params_grid, cv=self.cv)
        grid_search.fit(X, y)
        # Get the best model from the grid search
        best_lasso = grid_search.best_estimator_
        logger.info(f"_________________________________")
        logger.info('-- Lasso Regression Model Fit --')
        logger.info(f"Best regularization strength (L1): {grid_search.best_params_}")
        logger.info(f"Model weights: {best_lasso.coef_}")
        logger.info(f"Model bias: {best_lasso.intercept_}")
        logger.info(f"_________________________________")
        return best_lasso.coef_, best_lasso.intercept_

    def calculate_mad(self, X):
        """
        Calculate the Median Absolute Deviation (MAD) for each feature.
        """
        median_X = np.median(X, axis=0)
        mad = np.median(np.abs(X - median_X), axis=0)
        return mad

    def mad_distance(self, x_original, x_prime, mad):
        """
        Compute the MAD-normalized Manhattan distance between x and x'.
        """
        return np.sum(np.abs(x_original - x_prime) / (mad + 1e-8), axis=0)

    def objective(self, x_prime, x_original, y_prime, w, b, lambda_, mad):
        """
        Objective function for counterfactual optimization.
        """
        prediction = np.dot(x_prime, w) + b
        regression_loss = (prediction - y_prime) ** 2
        norm_manhattan_distance = self.mad_distance(x_prime, x_original, mad)
        total_loss = lambda_ * regression_loss + norm_manhattan_distance
        return total_loss

    def find_counterfactual(self, x_prime_initial, x_original, y_prime, w, b, epsilon, lambda_init=0.01):
        """
        Find the optimal counterfactual using optimization.
        """
        best_counterfactual = None
        best_lambda = None
        mad = self.calculate_mad(X)
        logger.info(f"Median Absolute Deviation (MAD): {mad}")
        prediction_final = np.dot(x_prime_initial, w) + b
        prediction_observed = np.dot(x_original, w) + b
        while (np.abs(prediction_final - y_prime) > epsilon) and (lambda_init < 1):
            result = minimize(self.objective, x_prime_initial, 
                                args=(x_original, y_prime, w, b, lambda_init, mad), 
                                method='Nelder-Mead')
            prediction_final = np.dot(result.x, w) + b
            best_counterfactual = result.x
            best_lambda = lambda_init
            lambda_init += 0.01
        logger.info(f"_________________________________")
        logger.info('-- Counterfactual Found --')
        logger.info(f"Counterfactual: {best_counterfactual}")
        logger.info(f"Optimal lambda: {best_lambda}")
        logger.info(f"_________________________________")
        logger.info(f"Original Input: {x_original} and Observed Output: {prediction_observed}")
        logger.info(f'Original Input: {x_original} and Desired Output: {y_prime}')
        logger.info(f'Counterfactual Input: {best_counterfactual} and Counterfactual Output: {prediction_final}')
        logger.info(f"_________________________________")
        return {
            'counterfactual': best_counterfactual,
            'optimal_lambda': best_lambda,
            'original_input': x_original,
            'observed_output': prediction_observed,
            'desired_output': y_prime,
            'counterfactual_output': prediction_final
        }

# Example data: features (X) and target (y)
X = np.random.normal(0, 1, (100, 2))
y = .2 * X[:, 0]  - .3 * X[:, 1] + np.random.normal(0, 10, 100)
# Define the range for regularization parameter (lambda)
param_grid = {'alpha': np.logspace(-4, 1, 6)}

# Initialize class
counterfactual_lasso = WachterCFLasso(params_grid=param_grid)

# Fit the Lasso model
w, b = counterfactual_lasso.fit(X, y)

# Set initial counterfactual guess
x_original = X[0]
x_prime_initial = X[0] + np.random.normal(0, 0.1, size=X[0].shape)
epsilon = 0.001
y_prime = -10

# Find the counterfactual
counterfactual = counterfactual_lasso.find_counterfactual(x_prime_initial, x_original, y_prime, w, b, epsilon)



