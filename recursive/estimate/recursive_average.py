import numpy as np

class RecursiveAverageFilter:
    """
    Recursive Average Filter algorithm for filtering out high frequency noise.
    """
    def __init__(self, recursive_mean: np.ndarray = None, num_iterations: int = 0) -> None:
        self.recursive_mean = recursive_mean
        self.num_iterations = num_iterations

    def update(self, observation: np.ndarray) -> None:
        " "
        self.num_iterations += 1
        if self.recursive_mean is None:
            self.recursive_mean = observation
        else:
            alpha = (self.num_iterations-1)/self.num_iterations  # alpha depends on the number of points, it is not free
            self.recursive_mean = alpha*self.recursive_mean + (1-alpha)*observation


if __name__=="__main__":
    
    # create some observations univariate
    observations = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    # create the model
    model = RecursiveAverageFilter()
    # update the model with each observation
    for observation in observations:
        model.update(observation)
    
    # print the final mean
    print(model.recursive_mean)
    # print batch mean
    print(np.mean(observations))