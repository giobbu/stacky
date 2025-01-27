import numpy as np

class KalmanFilter:
    """ Linear Kalman filter for state estimation. """
    def __init__(self, A : np.ndarray, H : np.ndarray, Q : np.ndarray, R : np.ndarray, x0 : np.ndarray, P0 : np.ndarray) -> None:
        self.A = A  # state transition matrix
        self.H = H  # observation matrix
        self.Q = Q  # state covariance
        self.R = R  # measurement covariance
        self.x = x0  # initial state
        self.P = P0  # initial state covariance

    def update(self, z : np.ndarray) -> None:
        " Update the state estimate with a new observation. "
        # Kalman filter update
        innovation = z - self.H @ self.x  # innovation
        innovation_cov = self.H @ self.P @ self.H.T + self.R  # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(innovation_cov)  # Kalman gain
        self.x = self.x + K @ innovation  # state update
        self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ self.P  # state covariance update

    def predict(self) -> tuple:
        " Predict the next state. "
        self.x = self.A @ self.x  # state prediction
        self.P = self.A @ self.P @ self.A.T + self.Q  # state covariance prediction
        return self.x, self.P
    

if __name__=='__main__':

    A = np.array([[1, 1], [0, 1]])  # state transition matrix
    H = np.array([[1, 0]])  # observation matrix
    Q = np.array([[0.1, 0], [0, 0.1]])  # state covariance
    R = np.array([[1]])  # measurement covariance
    x0 = np.array([[0], [0]])  # initial state
    P0 = np.array([[1, 0], [0, 1]])  # initial state covariance

    kf = KalmanFilter(A, H, Q, R, x0, P0)

    observations = np.random.normal(loc=0, scale=1, size=(1000, 1))
    list_predictions = []
    list_observed = []
    for z in observations:
        kf.update(z)
        pred, _ = kf.predict()
        list_predictions.append(pred[0][0])
        list_observed.append(z)

    import matplotlib.pyplot as plt
    figure = plt.figure(figsize=(10, 6))
    plt.plot(list_predictions, label='Predictions')
    plt.plot(list_observed, label='Observations', alpha=0.5)
    plt.legend()
    plt.show()