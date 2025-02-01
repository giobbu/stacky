import numpy as np

class DifferentialPrivacy:
    "Basic Differential Privacy with epsilon privacy budget and epsilon-delta privacy budget"
    def __init__(self, data, epsilon, mechanism='laplace', sensitivity=1, delta=0.01):
        assert isinstance(data, np.ndarray), "data must be a numpy array"
        self.data = data
        self.epsilon = epsilon
        self.mechanism = mechanism
        self.sensitivity = sensitivity
        self.delta = delta

    def gaussian_mechanism(self) -> np.ndarray:
        "Gaussian Mechanism"
        sigma = (self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta))) / self.epsilon
        noise = np.random.normal(0, sigma)  # Gaussian noise
        return self.data + noise
    
    def laplace_mechanism(self) -> np.ndarray:
        "Laplace Mechanism"
        scale = self.sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        return self.data + noise
    
    def anonymize(self, sensitivity=1, delta=0.01) -> np.ndarray:
        "Anonymize data using Laplace or Gaussian Mechanism"
        if self.mechanism == 'gaussian':
            return self.gaussian_mechanism()
        else:
            return self.laplace_mechanism()
    

if __name__== "__main__":
    # create data for testing
    data = np.random.randint(20, 60, 100)
    epsilon = 0.1
    mechanism = 'laplace'
    sensitivity = 1
    dp = DifferentialPrivacy(data=data, epsilon=epsilon, mechanism=mechanism, sensitivity=sensitivity)
    print(dp.anonymize())
