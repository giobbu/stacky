import numpy as np
from scipy import stats
from scipy.optimize import minimize


class DP_LogisticRegression:
    " Differentially private logistic regression"
    def __init__(self, X, y, lambda_reg=0.1):
        self.X = X
        self.y = y
        self.lambda_reg = lambda_reg
        self.n_samples, self.n_features = X.shape

    def logistic_regression(self):
        "Compute regularized logistic regression weights"

        def neg_log_likelihood(w):
            z = self.X @ w
            # Clipping to avoid overflow
            z = np.clip(z, -30, 30)
            log_likelihood = np.sum(self.y * z - np.log(1 + np.exp(z)))
            reg_term = (self.lambda_reg / 2) * np.sum(w ** 2)
            return -log_likelihood + reg_term

        def gradient(w):
            "Gradient of the negative log-likelihood"
            z = self.X @ w
            z = np.clip(z, -30, 30)
            sigmoid = 1 / (1 + np.exp(-z))
            grad = -self.X.T @ (self.y - sigmoid) + self.lambda_reg * w
            return grad
        # Initial weights
        w_init = np.zeros(self.n_features)
        # Minimize the negative log-likelihood
        result = minimize(neg_log_likelihood, w_init, jac=gradient, method='L-BFGS-B')
        return result.x
    
    def random_unit_vector(self, d):
        """Sample a random vector uniformly from the unit sphere in d dimensions"""
        # Generate d random normal variables
        vec = np.random.normal(size=d)
        return vec / np.linalg.norm(vec)
    
    def add_privacy_noise(self, w_star):
        " Add noise to the learned weights according to differential privacy requirements"
        # Calculate the scale parameter for the Gamma distribution: 2/(nλ)
        scale_param = 2 / (self.n_samples * self.lambda_reg)
        # Step 2.a: Sample the norm from Gamma(d, 2/(nλ)) distribution
        norm = np.random.gamma(shape=self.n_features, scale=scale_param)
        # Step 2.b: Sample a random direction uniformly
        direction = self.random_unit_vector(self.n_features)
        # Step 2.c: Combine magnitude and direction to create the noise vector η
        eta = norm * direction
        # Step 3: Add noise to the weights: w* + η
        return w_star + eta, eta


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic data
    n_samples = 1000  # This is 'n' from the mathematical formulation
    n_features = 10   # This is 'd' from the mathematical formulation
    
    print(f"Number of training examples (n): {n_samples}")
    print(f"Feature dimension (d): {n_features}")
    
    # True weights
    true_w = np.random.randn(n_features)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate binary labels
    logits = X @ true_w
    p = 1 / (1 + np.exp(-logits))
    y = np.random.binomial(1, p)
    
    # Set regularization parameter (λ)
    lambda_reg = 0.1
    print(f"Regularization parameter (λ): {lambda_reg}")
    
    # Step 1: Compute the non-private logistic regression weights w*
    print("\nStep 1: Training non-private logistic regression...")
    dp_lr = DP_LogisticRegression(X, y, lambda_reg)
    w_star = dp_lr.logistic_regression()
    
    # Step 2 and 3: Add privacy-preserving noise η
    print("\nStep 2 & 3: Adding differential privacy noise...")
    w_private, noise = dp_lr.add_privacy_noise(w_star)
    
    # Calculate the accuracy of both models
    def compute_accuracy(X, y, w):
        pred_probs = 1 / (1 + np.exp(-(X @ w)))
        predictions = (pred_probs >= 0.5).astype(int)
        return np.mean(predictions == y)
    
    non_private_acc = compute_accuracy(X, y, w_star)
    private_acc = compute_accuracy(X, y, w_private)
    
    print(f"\nNon-private accuracy: {non_private_acc:.4f}")
    print(f"Private accuracy: {private_acc:.4f}")
    
    # Calculate privacy budget (epsilon) - this is a simplified approximation
    # In practice, the privacy analysis would be more complex
    eps_approx = np.sqrt(8 * np.log(2/1e-5) / (n_samples * lambda_reg))
    print(f"\nApproximate privacy budget (ε): {eps_approx:.4f}")
    print(f"This means that the ratio of probabilities for neighboring datasets is bounded by e^{eps_approx:.4f}")