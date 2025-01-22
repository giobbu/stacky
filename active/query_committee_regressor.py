import numpy as np

class QueryCommitteeRegression:
    """
    Query by Committee for regression tasks, allowing active learning strategies
    like disagreement, random selection, and exploration-exploitation.
    """

    def __init__(self, X_train : np.ndarray, y_train : np.ndarray, n_queries : int, query_strategy : str, n_members : int=3, list_of_models: list =None) -> None:
        if list_of_models is None or len(list_of_models) != n_members:
            raise ValueError("Number of models must match n_members.")
        self.X_train = X_train
        self.y_train = y_train
        self.n_queries = n_queries
        self.query_strategy = query_strategy
        self.n_members = n_members
        self.list_of_models = list_of_models
        self.fitted_models = []

    def __repr__(self):
        return (
            f"QueryCommitteeRegression(n_queries={self.n_queries}, "
            f"query_strategy='{self.query_strategy}', n_members={self.n_members})"
        )

    def _fit_model(self, model: object, X: np.ndarray, y: np.ndarray) -> object:
        """Fit a single model on the training data."""
        return model.fit(X, y)

    def init_committee(self) -> None:
        """Initialize the committee by fitting all models on the training data."""
        self.fitted_models = [self._fit_model(model, self.X_train, self.y_train) for model in self.list_of_models]

    def _committee_predictions(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions from all committee members."""
        return np.array([model.predict(X) for model in self.fitted_models])

    def _committee_disagreement(self, X: np.ndarray) -> np.ndarray:
        """Compute the disagreement (variance) among the committee members' predictions."""
        predictions = self._committee_predictions(X)
        mean_predictions = predictions.mean(axis=0)
        disagreement = np.sum((predictions - mean_predictions) ** 2, axis=0)
        return disagreement

    def _select_query_idx(self, X: np.ndarray) -> int:
        """Select the query index based on the current query strategy."""
        if self.query_strategy == "disagree":
            return np.argmax(self._committee_disagreement(X))
        elif self.query_strategy == "random":
            return np.random.randint(0, X.shape[0])
        elif self.query_strategy == "exploration-exploitation":
            return (
                np.argmax(self._committee_disagreement(X))
                if np.random.rand() < 0.5
                else np.random.randint(0, X.shape[0])
            )
        else:
            raise ValueError("Invalid query strategy.")

    def committee_teach_selective(self, X_test: np.ndarray, y_test: np.ndarray) -> tuple:
        """
        Iteratively teach the committee by selecting informative points based on
        the query strategy, and retrain the committee with the updated training set.

        Returns:
            list_queries: List of queried points.
            list_insample_rmse: RMSE on the training set after each iteration.
            list_outsample_rmse: RMSE on the test set after each iteration.
            X_test: Remaining test features after queries.
            y_test: Remaining test labels after queries.
        """
        list_queries = []
        list_insample_rmse = []
        list_outsample_rmse = []

        for _ in range(self.n_queries):
            query_idx = self._select_query_idx(X_test)

            # Retrieve and store the queried point
            X_query = X_test[query_idx]
            y_query = y_test[query_idx]
            list_queries.append(X_query)

            # Update training data
            self.X_train = np.vstack((self.X_train, X_query))
            self.y_train = np.hstack((self.y_train, y_query))

            # Refit the committee
            self.init_committee()

            # Remove the queried point from the test set
            X_test = np.delete(X_test, query_idx, axis=0)
            y_test = np.delete(y_test, query_idx)

            # Calculate RMSE
            insample_predictions = self.committee_predict(self.X_train)[0]
            insample_rmse = np.sqrt(np.mean((self.y_train - insample_predictions) ** 2))

            outsample_predictions = self.committee_predict(X_test)[0]
            outsample_rmse = np.sqrt(np.mean((y_test - outsample_predictions) ** 2))

            list_insample_rmse.append(insample_rmse)
            list_outsample_rmse.append(outsample_rmse)

        return list_queries, list_insample_rmse, list_outsample_rmse, X_test, y_test

    def committee_predict(self, X: np.ndarray) -> tuple:
        """Predict using the committee: return mean and standard deviation of predictions."""
        predictions = self._committee_predictions(X)
        return predictions.mean(axis=0), predictions.std(axis=0)


    # def _bootstrap(self, X, y):
    #     pass

    # def _subsample(self, X, y):
    #     pass

    # def _pool():
    #     pass

    # def _selective(self):
    #     pass


if __name__=="__main__":

    # create univariate data
    np.random.seed(2)  # for reproducibility
    X = np.random.rand(2000, 1)

    # create target variable
    y = 2*X.squeeze() + np.random.normal(0, .1, 2000)

    # create non-linear data
    y = X.squeeze() + np.sin(100*X.squeeze()**3) + np.random.normal(0, .1, 2000)

    # select the first 100 points for training, and 100 points for querying/testing
    n_training_points = 30
    n_queries = 200

    # create training data
    X_train = X[:n_training_points]
    y_train = y[:n_training_points]

    # make predictions
    X_test = X[n_training_points:]
    y_test = y[n_training_points:]

    # create list of models of gaussian process with different kernels
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    models = [LinearRegression(), GradientBoostingRegressor()]
    n_members = len(models)

    # create QueryCommittee object
    qc = QueryCommitteeRegression(X_train, y_train, n_queries=15, query_strategy="disagree", n_members=n_members, list_of_models=models)
    qc.init_committee()

    # compute disagreement
    qc._committee_disagreement(X_test)

    # query the most informative points
    query_idx= qc.committee_query(X_test)
    X_query = X_test[query_idx]

    # plot the predictions and disagreement
    import matplotlib.pyplot as plt

    def rmse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred)**2))
    
    # compute the rmse for each model and the committee
    rmse_m1 = rmse(y_test, qc._committee_predictions(X_test)[0])
    rmse_m2 = rmse(y_test, qc._committee_predictions(X_test)[1])
    rmse_committee = rmse(y_test, qc.committee_predict(X_test)[0])

    figure = plt.figure(figsize=(12, 6))
    plt.plot(X_test, y_test, '*', label='True')
    plt.plot(X_test, qc._committee_predictions(X_test)[0], '+', label=f'M1 rmse={rmse_m1:.2f}')
    plt.plot(X_test, qc._committee_predictions(X_test)[1], 'x', label=f'M2 rmse={rmse_m2:.2f}')
    # plot the mean of the predictions
    plt.plot(X_test, qc.committee_predict(X_test)[0], '^', label=f'Committee Prediction rmse={rmse_committee:.2f}')
    # query point as vertical line
    plt.axvline(X_query, color='r', linestyle='--', label='Query Point')
    plt.legend()
    plt.show()

    # Cretaing a QueryCommittee object with random query strategy
    qc_r = QueryCommitteeRegression(X_train, y_train, n_queries=n_queries, query_strategy="random", n_members=n_members, list_of_models=models)
    qc_r.init_committee()
    qc_r.committee_teach(X_test, y_test)
    list_queries_r, list_insample_rmse_r, list_outsample_rmse_r, X_test_r, y_test_r = qc_r.committee_teach(X_test, y_test)

    # Create a QueryCommittee object with disagreement query strategy
    qc_d = QueryCommitteeRegression(X_train, y_train, n_queries=n_queries, query_strategy="disagree", n_members=n_members, list_of_models=models)
    qc_d.init_committee()
    qc_d.committee_teach(X_test, y_test)
    list_queries_d, list_insample_rmse_d, list_outsample_rmse_d, X_test_d, y_test_d = qc_d.committee_teach(X_test, y_test)

    # Create a QueryCommittee object with exploration-exploitation query strategy
    qc_e = QueryCommitteeRegression(X_train, y_train, n_queries=n_queries, query_strategy="exploration-exploitation", n_members=n_members, list_of_models=models)
    qc_e.init_committee()
    qc_e.committee_teach(X_test, y_test)
    list_queries_e, list_insample_rmse_e, list_outsample_rmse_e, X_test_e, y_test_e = qc_e.committee_teach(X_test, y_test)

    # plot the insample and outsample rmse
    figure = plt.figure(figsize=(12, 6))
    plt.plot(list_insample_rmse_r, label='Random Query Strategy')
    plt.plot(list_insample_rmse_d, label='Disagreement Query Strategy')
    plt.plot(list_insample_rmse_e, label='Exploration-Exploitation Query Strategy')
    plt.xlabel('Number of Queries')
    plt.ylabel('RMSE')
    plt.legend()
    plt.show()

    # plot the insample and outsample rmse
    figure = plt.figure(figsize=(12, 6))
    plt.plot(list_outsample_rmse_r, label='Random Query Strategy')
    plt.plot(list_outsample_rmse_d, label='Disagreement Query Strategy')
    plt.plot(list_outsample_rmse_e, label='Exploration-Exploitation Query Strategy')
    plt.xlabel('Number of Queries')
    plt.ylabel('RMSE')
    plt.legend()
    plt.show()

    # plot the queries
    figure = plt.figure(figsize=(12, 6))
    plt.plot(X_test, y_test, '*', label='True')
    plt.plot(list_queries_r, qc_r.committee_predict(list_queries_r)[0], 'o', label='Queries Random')
    plt.plot(list_queries_d, qc_d.committee_predict(list_queries_d)[0], 'x', label='Queries Disagree')
    plt.plot(list_queries_e, qc_e.committee_predict(list_queries_e)[0], '^', label='Queries Exploration-Exploitation')
    plt.legend()
    plt.show()

    # plot final predictions for both strategies
    figure = plt.figure(figsize=(12, 6))
    plt.plot(X_test, y_test, '*', label='True')
    plt.plot(X_test, qc_r.committee_predict(X_test)[0], 'o', label=f'Random Query Strategy rmse = {list_outsample_rmse_r[-1]:.2f}')
    plt.plot(X_test, qc_d.committee_predict(X_test)[0], '^', label=f'Disagreement Query Strategy rmse = {list_outsample_rmse_d[-1]:.2f}')
    plt.plot(X_test, qc_e.committee_predict(X_test)[0], 'x', label=f'Exploration-Exploitation Query Strategy rmse = {list_outsample_rmse_e[-1]:.2f}')
    plt.legend()
    plt.show()

