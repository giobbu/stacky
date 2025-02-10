from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import cvxpy as cp

class SyntheticControl(BaseEstimator, RegressorMixin):
    " A simple implementation of the synthetic control method using cvxpy"

    def __init__(self,):
        pass

    def fit(self, X, y):
        " Fit the model"
        X, y = check_X_y(X, y)  # check input data
        # Solve the optimization problem
        w = cp.Variable(X.shape[1])  # weights
        objective = cp.Minimize(cp.sum_squares(X@w - y))  # objective function
        constraints = [cp.sum(w) == 1, w >= 0]  # constraints
        problem = cp.Problem(objective, constraints)  # optimization problem
        problem.solve(verbose=False)  # solve the problem
        self.X_ = X
        self.y_ = y
        self.w_ = w.value
        self.is_fitted_ = True
        return self
        
    def predict(self, X):
        " Predict using the model"
        check_is_fitted(self)  # check if the model is fitted
        X = check_array(X)  # check input data
        return X @ self.w_


    



