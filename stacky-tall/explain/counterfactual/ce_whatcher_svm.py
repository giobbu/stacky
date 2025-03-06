import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from scipy.optimize import minimize
from loguru import logger

class WatcherCFSvm:
    def __init__(self, cv=2, params_grid=None):
        self.cv = cv
        self.params_grid = params_grid

    def fit(self, X, y):
        """
        Fit an SVR model to the data and return the model's support vectors, dual coefficients, and intercept.
        """
        svr = SVR()
        grid_search = GridSearchCV(svr, self.params_grid, cv=self.cv)
        grid_search.fit(X, y)

        best_svr = grid_search.best_estimator_
        logger.info(f"_________________________________")
        logger.info('-- SVR Model Fit --')
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Support Vectors: {best_svr.support_vectors_}")
        logger.info(f"Dual Coefficients: {best_svr.dual_coef_}")
        logger.info(f"Intercept: {best_svr.intercept_}")
        logger.info(f"_________________________________")

        return best_svr

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

    def kernel_function(self, X1, X2, kernel, **params):
        """
        Compute the kernel function manually.
        """
        if kernel == 'linear':
            return np.dot(X1, X2.T)
        elif kernel == 'poly':
            return (params.get('gamma', 1) * np.dot(X1, X2.T) + params.get('coef0', 1)) ** params.get('degree', 3)
        elif kernel == 'rbf':
            gamma = params.get('gamma', 1)
            squared_distances = np.sum(X1**2, axis=1, keepdims=True) - 2 * np.dot(X1, X2.T) + np.sum(X2**2, axis=1)
            return np.exp(-gamma * squared_distances)
        else:
            raise ValueError("Unsupported kernel")

    def objective(self, x_prime, x_original, y_prime, svr_model, lambda_, mad):
        """
        Objective function for counterfactual optimization.
        """
        support_vectors = svr_model.support_vectors_
        dual_coefs = svr_model.dual_coef_
        intercept = svr_model.intercept_[0]
        kernel = svr_model.kernel
        params = {'gamma': svr_model._gamma, 'coef0': svr_model.coef0, 'degree': svr_model.degree}

        kernel_values = self.kernel_function(support_vectors, x_prime.reshape(1, -1), kernel, **params)
        prediction = np.sum(dual_coefs @ kernel_values) + intercept

        regression_loss = (prediction - y_prime) ** 2
        norm_manhattan_distance = self.mad_distance(x_prime, x_original, mad)
        total_loss = lambda_ * regression_loss + norm_manhattan_distance

        return total_loss

    def find_counterfactual(self, x_prime_initial, x_original, y_prime, svr_model, epsilon, lambda_init=0.01):
        """
        Find the optimal counterfactual using optimization.
        """
        best_counterfactual = None
        best_lambda = None
        mad = self.calculate_mad(X)
        logger.info(f"Median Absolute Deviation (MAD): {mad}")

        prediction_final = svr_model.predict([x_prime_initial])[0]
        prediction_observed = svr_model.predict([x_original])[0]

        while (np.abs(prediction_final - y_prime) > epsilon) and (lambda_init < 1):
            result = minimize(self.objective, x_prime_initial, 
                              args=(x_original, y_prime, svr_model, lambda_init, mad), 
                              method='Nelder-Mead')

            prediction_final = svr_model.predict([result.x])[0]
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

# Example Data: Features (X) and Target (y)
X = np.random.normal(0, 1, (100, 2))
y = 0.2 * X[:, 0] - 0.3 * X[:, 1] + np.random.normal(0, 10, 100)

# Define the grid for SVR hyperparameters (with tolerance)
param_grid = {
    'kernel': ['linear', 'poly', 'rbf'],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3],
    'coef0': [0.0, 1.0],
    'tol': [1e-3, 1e-4]  # Added tolerance as stopping criterion
}

# Initialize the CounterfactualOptimizationSVM class
counterfactual_svm = CounterfactualOptimizationSVM(params_grid=param_grid)

# Fit the SVM model
svr_model = counterfactual_svm.fit(X, y)

# Set initial counterfactual guess
x_original = X[0]
x_prime_initial = X[0] + np.random.normal(0, 0.1, size=X[0].shape)
epsilon = 0.001
y_prime = -10

# Find the counterfactual
counterfactual = counterfactual_svm.find_counterfactual(x_prime_initial, x_original, y_prime, svr_model, epsilon)
