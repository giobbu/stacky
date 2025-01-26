import numpy as np

class PermutationMLScore:
    def __init__(self, model, X, y, metric, n_iter=1000):
        self.model = model
        self.X = X
        self.y = y
        self.metric = metric
        self.n_iter = n_iter

    def permutation_test(self):
        baseline_score = self.metric(self.y, self.model.predict(self.X))
        null_scores = np.zeros(self.n_iter)
        for i in range(self.n_iter):
            X_permuted = self.X.copy()
            X_permuted = X_permuted.apply(np.random.permutation)
            null_scores[i] = self.metric(self.y, self.model.predict(X_permuted))
        return null_scores, baseline_score
    
    def compute_pvalue(self, null_scores, baseline_score):
        return np.sum(null_scores <= baseline_score) / self.n_iter

    def permutation_score(self):
        null_scores, baseline_score = self.permutation_test()
        pvalue = self.compute_pvalue(null_scores, baseline_score)
        return baseline_score, null_scores, pvalue
    
if __name__=='__main__':
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    # create a regression dataset
    df = pd.DataFrame(np.random.rand(1000, 5))
    df.columns = ['A', 'B', 'C', 'D', 'E']
    df['target'] = np.sin(df['A'] + 2*df['B'])**4 * np.cos( 3*df['C'] * 4*df['D']) * 5*df['E']**2 + np.random.normal(10, 5, 1000)
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    pmls = PermutationMLScore(rf, X_test, y_test, mean_squared_error)
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
    pmls = PermutationMLScore(lr, X_test, y_test, mean_squared_error)
    baseline_score, null_scores, pvalue = pmls.permutation_score()
    print(f'Baseline score: {baseline_score}')
    print(f'P-value: {pvalue}')
    plt.hist(null_scores, bins=30)
    plt.axvline(x=baseline_score, color='red', linestyle='--')
    plt.xlabel('Mean Squared Error')
    plt.ylabel('Frequency')
    plt.title('Permutation Test Score')
    plt.show()
