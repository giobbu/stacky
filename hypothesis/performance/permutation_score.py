import numpy as np

import numpy as np

class PermutationRegressorScore:
    """Permutation test for regression models."""
    
    def __init__(self, model, X, y, metric, n_iter=1000):
        """
        Initialize the PermutationRegressorScore.

        :param model: The regression model to evaluate.
        :param X: The feature matrix.
        :param y: The target vector.
        :param metric: The evaluation metric function.
        :param n_iter: Number of permutation iterations (default: 1000).
        """
        self.model = model
        self.X = X
        self.y = y
        self.metric = metric
        self.n_iter = n_iter

    def _permutation_test(self):
        """Compute null scores and baseline score."""
        baseline_score = self.metric(self.y, self.model.predict(self.X))  # Baseline score
        null_scores = np.zeros(self.n_iter)
        
        for i in range(self.n_iter):
            y_permuted = np.random.permutation(self.y)  # Permute target
            null_scores[i] = self.metric(y_permuted, self.model.predict(self.X))  # Null score
        
        return null_scores, baseline_score

    def _compute_pvalue(self, null_scores, baseline_score):
        """Compute p-value."""
        return np.mean(null_scores <= baseline_score)

    def permutation_score(self):
        """Compute permutation test score."""
        null_scores, baseline_score = self._permutation_test()
        pvalue = self._compute_pvalue(null_scores, baseline_score)
        return baseline_score, null_scores, pvalue
    
if __name__=='__main__':
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    # create a regression dataset
    df = pd.DataFrame(np.random.rand(10000, 5))
    df.columns = ['A', 'B', 'C', 'D', 'E']
    df['target'] = np.sin(df['A'] + 2*df['B'])**4 * np.cos( 3*df['C'] * 4*df['D']) * 5*df['E']**2 + np.random.normal(0, 5, 10000)
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    pmls = PermutationRegressorScore(rf, X_test, y_test, mean_squared_error)
    baseline_score, null_scores, pvalue = pmls.permutation_score()
    print(f'Baseline score: {baseline_score}')
    print(f'P-value: {pvalue}')

    # print histogram of null scores and vertical line for baseline score
    import matplotlib.pyplot as plt
    plt.hist(null_scores, bins=30)
    plt.axvline(x=baseline_score, color='red', linestyle='--')
    plt.xlabel('Mean Squared Error')
    plt.ylabel('Frequency')
    plt.title('Permutation Test Score')
    plt.show()

    # compare with linear regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pmls = PermutationRegressorScore(lr, X_test, y_test, mean_squared_error)
    baseline_score, null_scores, pvalue = pmls.permutation_score()
    print(f'Baseline score: {baseline_score}')
    print(f'P-value: {pvalue}')
    plt.hist(null_scores, bins=30)
    plt.axvline(x=baseline_score, color='red', linestyle='--')
    plt.xlabel('Mean Squared Error')
    plt.ylabel('Frequency')
    plt.title('Permutation Test Score')
    plt.show()
