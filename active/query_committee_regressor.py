import numpy as np

class QueryCommitteeRegression:
    " Query by committee for regression. "

    def __init__(self, X_train, y_train, n_queries, query_strategy, n_members=3, list_of_models=None):
        assert len(list_of_models) == n_members, "Number of models must be equal to n_members."
        self.X_train = X_train
        self.y_train = y_train
        self.n_queries = n_queries
        self.query_strategy = query_strategy
        self.n_members = n_members,
        self.list_of_models = list_of_models
    
    def __repr__(self):
        return f"QueryByCommittee(n_queries={self.n_queries}, query_strategy={self.query_strategy}, n_members={self.n_members})"
    
    def _fit_models(self, model, X, y):
        " Fit the models on the training data. "
        fitted_model = model.fit(X, y)
        return fitted_model

    def init_committee(self):
        " Initialize the committee of models. "
        self.fitted_models = []
        for model in self.list_of_models:
            fitted_model = self._fit_models(model, self.X_train, self.y_train)
            self.fitted_models.append(fitted_model)

    def _committee_predictions(self, X):
        " Make predictions using the committee of models. "
        predictions = []
        for model in self.fitted_models:
            prediction = model.predict(X)
            predictions.append(prediction)
        return np.array(predictions)
            
    def _committee_disagreement(self, X):
        " Compute the disagreement among the committee members. "
        predictions = self._committee_predictions(X)
        mean_predictions = predictions.mean(axis=0)
        ambiguity = np.sum((predictions - mean_predictions)**2, axis=0)
        return ambiguity
    
    def committee_query(self, X):
        " Query the most informative points. "
        disagreement = self._committee_disagreement(X)
        # get the indices to query the most informative point
        query_idx = np.argsort(disagreement)[-1]
        return query_idx
    
    def committee_teach(self, X_test, y_test):
        " Teach the committee the new data."
        list_queries = []
        list_insample_rmse = []
        list_outsample_rmse = []
        for query in range(self.n_queries):
            if self.query_strategy == "disagree":
                query_idx = self.committee_query(X_test)
            elif self.query_strategy == "random":
                query_idx = np.random.randint(0, X_test.shape[0])
            elif self.query_strategy == "exploration-exploitation":
                # widraw a number between 0 and 1
                p = np.random.rand()
                if p < 0.5:
                    query_idx = self.committee_query(X_test)
                else:
                    query_idx = np.random.randint(0, X_test.shape[0])
            else:
                raise ValueError("Invalid query strategy.")
            # get the query point
            X_query = X_test[query_idx]
            y_query = y_test[query_idx]
            # add the query point to the training data
            self.X_train = np.vstack((self.X_train, X_query))
            self.y_train = np.hstack((self.y_train, y_query))
            # refit the models
            self.init_committee()
            # remove the query point from the test data
            X_test = np.delete(X_test, query_idx, axis=0)
            y_test = np.delete(y_test, query_idx)
            # store the query point
            list_queries.append(X_query)
            # compute insample and outsample rmse
            insample_predictions = self.committee_predict(self.X_train)[0]
            insample_rmse = np.sqrt(np.mean((self.y_train - insample_predictions)**2))
            outsample_predictions = self.committee_predict(X_test)[0]

            outsample_rmse = np.sqrt(np.mean((y_test - outsample_predictions)**2))
            list_insample_rmse.append(insample_rmse)
            list_outsample_rmse.append(outsample_rmse)

        return list_queries, list_insample_rmse, list_outsample_rmse, X_test, y_test
    
    def committee_predict(self, X):
        " Make predictions using the committee of models. "
        predictions = self._committee_predictions(X)
        return predictions.mean(axis=0), predictions.std(axis=0)



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

