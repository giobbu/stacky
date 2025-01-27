import numpy as np

class KalmanFilter:

    def __init__(self, A, H, Q, R, x0, P0):
        self.A = A  # state transition matrix
        self.H = H  # observation matrix
        self.Q = Q  # state covariance
        self.R = R  # measurement covariance
        self.x = x0  # initial state
        self.P = P0  # initial state covariance

    def update(self, z):
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)  # Kalman gain
        self.x = self.x + K @ (z - self.H @ self.x)  # state update
        self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ self.P  # state covariance update

    def predict(self):
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