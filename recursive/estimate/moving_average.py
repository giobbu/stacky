import numpy as np

class MovingAverageFilter:
    " Moving Average Filter algorithm for filtering out high frequency noise. "
    def __init__(self,moving_mean: np.ndarray = None, num_iterations: int = 0, window: int = 3) -> None:
        self.moving_mean = moving_mean
        self.num_iterations = num_iterations
        self.window = window
        self.observations = []

    def update(self, observation: np.ndarray) -> None:
        " Update the model with the new observation. "
        self.num_iterations += 1
        if (self.moving_mean==None) and (self.num_iterations < self.window):
            self.moving_mean=observation
            self.observations.append(observation)
        elif (self.num_iterations < self.window):
            alpha = (self.num_iterations-1)/self.num_iterations  # alpha depends on the number of points, it is not free
            self.moving_mean = alpha*self.moving_mean + (1-alpha)*observation
            self.observations.append(observation)
        else:
            self.observations.append(observation)
            self.moving_mean = self.moving_mean + (self.observations[-1] - self.observations[0])/self.window
            del self.observations[0]
            

if __name__=="__main__":
    # create univariate data
    observations = np.random.normal(5, 1, 1000)
    
    # instanciate model
    model = MovingAverageFilter(window=10)
    # update the model with each observation
    list_moving_avg = []
    for observation in observations:
        model.update(observation)
        list_moving_avg.append(model.moving_mean)
    
    # plot results displaying data and moving average
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5))
    plt.plot(observations)
    plt.plot(list_moving_avg)
    plt.show()

