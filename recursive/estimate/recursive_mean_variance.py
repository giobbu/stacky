import numpy as np

class RecursiveMeanVariance:
    """
    Recursive Mean and Variance for large datasets.
    """
    def __init__(self, recursive_mean: np.ndarray = None, recursive_variance: np.ndarray = None, num_iterations: int = 0) -> None:
        self.recursive_mean = recursive_mean
        self.recursive_variance = recursive_variance
        self.num_iterations = num_iterations

    def _update_mean(self) -> None:
        " "
        if self.recursive_mean is None:
            self.recursive_mean = observation
        else:
            self.recursive_mean_1 = self.recursive_mean
            self.recursive_mean = self.recursive_mean_1 + (observation - self.recursive_mean_1)/self.num_iterations

    def _update_variance(self) -> None:
        " "
        if self.recursive_variance is None:
            self.recursive_variance = 0
        else:
            self.recursive_variance_1 = self.recursive_variance
            self.recursive_variance = self.recursive_variance_1 + self.recursive_mean_1**2 - self.recursive_mean**2 + (observation**2 - self.recursive_variance_1 - self.recursive_mean_1**2)/self.num_iterations

    def update(self, observation: np.ndarray) -> None:
        " "
        self.num_iterations += 1
        if self.recursive_mean is None:
            self.recursive_mean = observation
            self.recursive_variance = 0
        else:
            self._update_mean()
            self._update_variance()
        return self.recursive_mean, self.recursive_variance



if __name__=="__main__":
    
    # create some observations univariate
    observations = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    # create the model
    model = RecursiveMeanVariance()
    # update the model with each observation
    for observation in observations:
        model.update(observation)
    print('-'*20)
    # print the final mean
    print('recursive mean', model.recursive_mean)
    # print batch mean
    print('batch mean', np.mean(observations))
    print('-'*20)
    # print the final std
    print('recursive variance', model.recursive_variance)
    # print batch std
    print('batch variance', np.var(observations))
    print('-'*20)