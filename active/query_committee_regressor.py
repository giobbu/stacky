import numpy as np

class QueryCommitteeRegression:
    " Query by committee for regression. "

    def __init__(self, X_train, y_train, n_queries, query_strategy, n_members=3):
        self.X_train = X_train
        self.y_train = y_train
        self.n_queries = n_queries
        self.query_strategy = query_strategy
        self.n_members = n_members
    
    def __repr__(self):
        return f"QueryByCommittee(n_queries={self.n_queries}, query_strategy={self.query_strategy}, n_members={self.n_members})"
    
    def _fit_models(self, model, X, y):
        " Fit the models on the training data. "
        fitted_model = model.fit(X, y)
        return fitted_model

    def init_committee(self, list_of_models):
        " Initialize the committee of models. "
        assert len(list_of_models) == self.n_members, "Number of models must be equal to n_members."
        self.fitted_models = []
        for model in list_of_models:
            fitted_model = self._fit_models(model, self.X_train, self.y_train)
            self.fitted_models.append(fitted_model)

    def _committee_predict(self, X):
        " Make predictions using the committee of models. "
        predictions = []
        for model in self.fitted_models:
            prediction = model.predict(X)
            predictions.append(prediction)
        return np.array(predictions)
    
    def _committee_disagreement(self, X):
        " Compute the disagreement among the committee members. "
        predictions = self._committee_predict(X)
        disagreement = np.std(predictions, axis=0)
        return disagreement
    
    def query(self, X):
        " Query the most informative points. "
        if self.query_strategy == "disagreement":
            disagreement = self._committee_disagreement(X)
            query_idx = np.argsort(disagreement)[-self.n_queries:]
        return query_idx, X[query_idx]



if __name__=="__main__":
    # create univariate data
    np.random.seed(42)
    X = np.random.rand(100, 1)

    # create target variable
    y = 2*X.squeeze() + np.random.normal(0, 1, 100)

    # create non-linear data
    y = np.sin(2*np.pi*X.squeeze()) + np.random.normal(0, 0.1, 100)

    # create training data
    X_train = X[:10]
    y_train = y[:10]

    # create list of models
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    models = [LinearRegression(), LinearRegression(fit_intercept=False), RandomForestRegressor()]
    n_members = len(models)

    # create QueryCommittee object
    qc = QueryCommitteeRegression(X_train, y_train, n_queries=15, query_strategy="disagreement", n_members=n_members)
    qc.init_committee(models)

    # make predictions
    X_test = X[10:]
    y_test = y[10:]
    print(qc._committee_predict(X_test).shape)

    # compute disagreement
    qc._committee_disagreement(X_test)

    # query the most informative points
    query_idx, X_query = qc.query(X_test)

    # plot the predictions and disagreement
    import matplotlib.pyplot as plt
    plt.plot(X_test, y_test, '*', label='True')
    plt.plot(X_test, qc._committee_predict(X_test)[0], 'o', label='Linear Regression')
    plt.plot(X_test, qc._committee_predict(X_test)[1], '-', label='Linear Regression (No Intercept)')
    plt.plot(X_test, qc._committee_predict(X_test)[2], '+', label='Random Forest')
    #plt.plot(X_test, qc._committee_disagreement(X_test), '*', label='Disagreement')
    # query point as vertical line
    for idx in query_idx:
        plt.axvline(X_test[idx], color='red', linestyle='--')
    plt.legend()
    plt.show()