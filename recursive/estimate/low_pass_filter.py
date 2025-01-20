import numpy as np

class LowPassFilter:
    """
    Low Pass Filter algorithm for filtering out high frequency noise.
    """
    def __init__(self, lowpass_mean: np.ndarray = None, num_iterations: int = 0, alpha: float = 0.999) -> None:
        self.lowpass_mean = lowpass_mean
        self.num_iterations = num_iterations
        self.alpha = alpha

    def update(self, observation: np.ndarray) -> None:
        " "
        self.num_iterations += 1
        if self.lowpass_mean is None:
            self.lowpass_mean = observation
        else:
            self.lowpass_mean = self.alpha*self.lowpass_mean + (1-self.alpha)*observation


if __name__=="__main__":
    
    # create some observations univariate
    observations = np.random.normal(loc=0, scale=1, size=1000)
    
    # create the model
    model = LowPassFilter(alpha=0.99)
    # update the model with each observation
    list_lowpass_avg = []
    for observation in observations:
        model.update(observation)
        list_lowpass_avg.append(model.lowpass_mean)
    
    # plot results displaying data and moving average
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5))
    plt.plot(observations)
    plt.plot(list_lowpass_avg)
    plt.show()