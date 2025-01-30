import numpy as np

class RecursiveLeastSquaresFilter:
    """
    Recursive Least Squares Filter algorithm for adaptive learning.
    """

    def __init__(self, num_variables: int, forgetting_factor: float, initial_delta: float) -> None:
        """
        Initialize the RecurrentLeastSquares object.

        Parameters:
        - num_variables: int, number of variables including the constant
        - forgetting_factor: float, forgetting factor (lambda), usually close to 1
        - initial_delta: float, controls the initial state
        """

        if num_variables <= 0:
            raise ValueError("Number of variables must be positive.")
        if forgetting_factor <= 0:
            raise ValueError("Forgetting factor must be positive.")
        if initial_delta <= 0:
            raise ValueError("Initial delta must be positive.")

        self.num_variables = num_variables
        self.A = initial_delta * np.identity(self.num_variables)
        self.w = np.zeros((self.num_variables, 1))
        self.forgetting_factor_inverse = 1 / forgetting_factor
        self.num_observations = 1
        self.residual = np.array([0.0]).reshape(1, -1)
        self.residual_sqr = np.array([0.0]).reshape(1, -1)
        self.rmse = np.array([0.0]).reshape(1, -1)

    def update(self, observation: np.ndarray, label: np.ndarray) -> None:
        """
        Update the model with a new observation and label.

        Parameters:
        - observation: numpy array, observation vector
        - label: float, true label corresponding to the observation
        """
        z = self.forgetting_factor_inverse * self.A @ observation
        alpha = 1 / (1 + observation.T @ z)
        self.w += (label - alpha * observation.T @ (self.w + label * z)) * z
        self.A -= alpha * z @ z.T
        self.num_observations += 1
        self.residual = label - self.w.T @ observation
        self.residual_sqr = (label - self.w.T @ observation) ** 2

    def fit(self, observations: np.ndarray, labels: np.ndarray) -> None:
        """
        Fit the model to a set of observations and labels.

        Parameters:
        - observations: list of numpy arrays, each array representing an observation vector
        - labels: list of floats, true labels corresponding to the observations
        """
        if len(observations) != len(labels):
            raise ValueError("Number of observations must be equal to the number of labels.")

        for i in range(len(observations)):
            observation = np.transpose(np.matrix(observations[i]))
            self.update(observation, labels[i])

    def predict(self, observation: np.ndarray) -> float:
        """
        Predict the value of a new observation.

        Parameters:
        - observation: numpy array, observation vector
        Returns:
        - float, predicted value based on the observation
        """
        return np.dot(self.w.T, observation).item() #float(self.w.T @ observation)



if __name__ == "__main__":

    # Set number of lags
    num_lags = 2
    # Forgetting factor
    forgetting_factor = 0.99
    # Initial delta
    initial_delta = 0.1

    # Initialize model
    model = RecursiveLeastSquaresFilter(num_variables=num_lags, 
                                    forgetting_factor=forgetting_factor, 
                                    initial_delta=initial_delta)
    
    # Generate observations (univariate time series)
    np.random.seed(0)
    n = 1000
    observations = np.random.normal(size=n)

    # Store predictions and observed values
    list_predictions = []
    list_observed = []
    list_residuals = []

    # loop through the observations
    for i in range(num_lags, n):

        if i >= len(observations) - num_lags - 1:
            break

        # Get the input
        X = observations[i:i + num_lags].reshape(-1, 1)
        
        # Make a prediction
        y_pred = model.predict(X)

        # Get the true value
        y = observations[i + num_lags + 1].reshape(-1, 1)

        # Update the model and make a prediction
        model.update(X, y)

        # Store prediction and observed values
        list_predictions.append(y_pred)
        list_observed.append(y[0])

        # Compute residuals
        residuals = np.abs(y - y_pred)
        list_residuals.append(residuals[0])

    # Compute the root mean squared error
    rmse = np.sqrt(np.mean(np.array(list_residuals) ** 2))
    print(f"Root Mean Squared Error: {rmse}")

    # Plot predictions and observations
    # plot results displaying data and moving average
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5))
    plt.plot(list_observed)
    plt.plot(list_predictions)
    plt.show()

