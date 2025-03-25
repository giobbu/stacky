import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def entropy_loss(w):
    """Objective function: entropy loss H(w) = sum(w_i * log(w_i))"""
    return np.sum(w * np.log(w))

def constraint_balance(w, X_treated, X_control):
    """Ensures Σ w_i * ̃X_i = 0"""
    treated_mean = np.mean(X_treated, axis=0)
    X_tilde = X_control - treated_mean
    return np.dot(w, X_tilde)

def constraint_sum(w):
    """Ensures Σ w_i = 1"""
    return np.sum(w) - 1

def entropy_balancing(X, T):
    """
    Solves for entropy balancing weights w.
    """
    treated = T > 0
    control = T == 0
    
    X_treated = X[treated]
    X_control = X[control]
    
    n_control = X_control.shape[0]
    
    # Initial uniform weights
    w0 = np.ones(n_control) / n_control
    
    # Constraints
    constraints = [
        {"type": "eq", "fun": constraint_balance, "args": (X_treated, X_control)},
        {"type": "eq", "fun": constraint_sum}
    ]
    
    # Bounds (weights must be positive)
    bounds = [(0, None)] * n_control

    # Solve optimization
    result = minimize(entropy_loss, w0, constraints=constraints, bounds=bounds, method="SLSQP")
    
    if result.success:
        return result.x
    else:
        raise ValueError("Optimization failed:", result.message)

def compute_smd(X_treated, X_control, weights=None):
    """
    Compute Standardized Mean Difference (SMD) between treated and control groups.
    """
    treated_mean = np.mean(X_treated, axis=0)
    
    if weights is None:
        control_mean = np.mean(X_control, axis=0)
        control_var = np.var(X_control, axis=0, ddof=1)
    else:
        control_mean = np.average(X_control, axis=0, weights=weights)
        control_var = np.average((X_control - control_mean) ** 2, axis=0, weights=weights)
    
    treated_var = np.var(X_treated, axis=0, ddof=1)
    
    pooled_std = np.sqrt((treated_var + control_var) / 2)
    
    smd = (treated_mean - control_mean) / pooled_std
    return smd

def plot_smd(smd_before, smd_after):
    """Plots SMD before and after weighting."""
    n_features = len(smd_before)
    indices = np.arange(n_features)
    
    plt.figure(figsize=(8, 5))
    width = 0.35  # Bar width

    plt.bar(indices - width/2, smd_before, width, label="Before Weighting", color="red", alpha=0.7)
    plt.bar(indices + width/2, smd_after, width, label="After Weighting", color="blue", alpha=0.7)

    plt.axhline(y=0.1, color="gray", linestyle="dashed", linewidth=1)  # Balance threshold
    plt.axhline(y=-0.1, color="gray", linestyle="dashed", linewidth=1)

    plt.xlabel("Covariates")
    plt.ylabel("Standardized Mean Difference (SMD)")
    plt.title("SMD Before and After Entropy Balancing")
    plt.xticks(indices, [f"X{i+1}" for i in range(n_features)])
    plt.legend()
    plt.show()

# Example usage
np.random.seed(42)
n_samples = 100
n_features = 5

X = np.random.randn(n_samples, n_features)  # Random covariate matrix
T = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])  # 30% treated, 70% control

treated = T > 0
control = T == 0

X_treated = X[treated]
X_control = X[control]

# Compute SMD before weighting
smd_before = compute_smd(X_treated, X_control)

# Compute entropy balancing weights
weights = entropy_balancing(X, T)

# Compute SMD after weighting
smd_after = compute_smd(X_treated, X_control, weights)

# Plot SMD comparison
plot_smd(smd_before, smd_after)