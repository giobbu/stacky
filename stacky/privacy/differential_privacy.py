import numpy as np

class EpsDiffPrivacy:
    "Differential Privacy Algorithm for Privacy Preserving Data Publishing with Epsilon-Differential Privacy"
    def __init__(self, data, epsilon, sensitivity=1):
        assert isinstance(data, np.ndarray), "data must be a numpy array"
        self.data = data
        self.epsilon = epsilon
        self.sensitivity = sensitivity

    def laplace_mechanism(self, data):
        return data + np.random.laplace(loc=0, scale=self.sensitivity/self.epsilon, size=data.shape)

    def anonymize(self):
        return self.laplace_mechanism(self.data)
    

if __name__== "__main__":
    # create data for testing
    data = np.random.randint(20, 60, 100)
    epsilon = 1
    eps_diff_privacy = EpsDiffPrivacy(data, epsilon)
    anonymized_data = eps_diff_privacy.anonymize()
    print(data)
    print('--'*20)
    print(anonymized_data)