import numpy as np

class RecurrentMean:
    """
    Recurrent Least Squares algorithm for adaptive learning.
    """
    def __init__(self, current_mean: np.ndarray = None, num_iterations: int = 0) -> None:
        """
        Initialize the RecurrentMean object.

        Parameters:
        - current_mean: np.ndarray, current mean
        - num_iterations: int, number of iterations
        """
        self.current_mean = current_mean
        self.num_iterations = num_iterations


    def update(self, observation: np.ndarray) -> None:
        if self.current_mean is None:
            self.current_mean = observation
        else:
            self.current_mean = self.current_mean + (observation - self.current_mean) / (self.num_iterations + 1)
        self.num_iterations += 1

if __name__=="__main__":
    # create some observations univariate
    observations = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # create the model
    model = RecurrentMean()
    # update the model with each observation
    for observation in observations:
        model.update(observation)
    # print the final mean
    print(model.current_mean)
    # print batch mean
    print(np.mean(observations))