from sklearn.utils.validation import check_array, check_is_fitted
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

class VanillaDoublyRobustEstimator:
    """A simple implementation of the doubly robust estimator."""
    
    def __init__(self, df: pd.DataFrame, X: list, T: str, Y: str, expectation_model, propensity_model):
        self.df = df
        self.X = X  # Features (should be a list of column names)
        self.T = T  # Treatment
        self.Y = Y  # Outcome
        self.expectation_model = expectation_model .__class__() # Separate instance for mu1
        self.propensity_model = propensity_model  # Propensity model
        
    def fit(self):
        """Fit all models."""
        self.expectation_model_mu1 = self.expectation_model.__class__()
        self.expectation_model_mu0 = self.expectation_model.__class__()
        self.propensity_model.fit(self.df[self.X], self.df[self.T])
        self.expectation_model_mu1.fit(self.df.query(f"{self.T} == 1")[self.X], self.df.query(f"{self.T} == 1")[self.Y])
        self.expectation_model_mu0.fit(self.df.query(f"{self.T} == 0")[self.X], self.df.query(f"{self.T} == 0")[self.Y])

    def predict(self):
        """Predict mu0, mu1, and ps."""
        ps = self.propensity_model.predict_proba(self.df[self.X])[:, 1]
        mu0 = self.expectation_model_mu0.predict(self.df[self.X])
        mu1 = self.expectation_model_mu1.predict(self.df[self.X])
        return ps, mu0, mu1

    def estimate_ATE(self):
        """Estimate the average treatment effect."""
        ps, mu0, mu1 = self.predict()
        ATE_hat = (np.mean(self.df[self.T] * (self.df[self.Y] - mu1) / ps + mu1) -
                   np.mean((1 - self.df[self.T]) * (self.df[self.Y] - mu0) / (1 - ps) + mu0))
        return ATE_hat
    
    def bootstrap(self, n_iter=1000):
        """Bootstrap the ATE."""
        ATE_hats = Parallel(n_jobs=-1)(delayed(self.bootstrap_single)() for _ in range(n_iter))
        return ATE_hats
    
    def bootstrap_single(self):
        """Bootstrap a single iteration."""
        idx = np.random.choice(self.df.index, len(self.df), replace=True)
        df_boot = self.df.loc[idx]
        dr_boot = VanillaDoublyRobustEstimator(df_boot, self.X, self.T, self.Y, self.expectation_model, self.propensity_model)
        dr_boot.fit()
        return dr_boot.estimate_ATE()

if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'age': np.random.randint(20, 60, 100),
        'income': np.random.randint(1, 100, 100),
        'treatment': np.random.randint(0, 2, 100),
        'outcome': np.random.randint(1000, 10000, 100)
    })
    
    X = ['age', 'income']  # Features should be a list
    T = 'treatment'
    Y = 'outcome'
    
    from sklearn.linear_model import LinearRegression, LogisticRegression
    
    doubly_robust_estimator = VanillaDoublyRobustEstimator(data, X, T, Y, LinearRegression(), LogisticRegression())
    doubly_robust_estimator.fit()
    
    ATE_hat = doubly_robust_estimator.estimate_ATE()
    print(f"Estimated ATE: {ATE_hat}")

    ATE_hats = doubly_robust_estimator.bootstrap(n_iter=1000)
    # print 95% confidence interval
    print(f"95% confidence interval: [{np.percentile(ATE_hats, 2.5)}, {np.percentile(ATE_hats, 97.5)}]")
