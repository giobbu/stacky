from vanilla_dtw import VanillaDTW
from window_dtw import WindowDTW
import numpy as np

class KNN_DTW:
    "K-Nearest Neighbors using Dynamic Time Warping (DTW) as the similarity measure."
    def __init__(self, dataset: list, k: int, names: list):
        assert k > 0, "k must be greater than 0."
        assert names is not None, "names must be provided."
        assert len(dataset) == len(names), "The number of names must be equal to the number of time series."
        assert type(dataset) == list, "dataset must be a list."
        assert all([type(ts) == list for ts in dataset]), "Each time series must be a list."
        assert all([len(ts) > 2 for ts in dataset]), "Each time series must have at least one element."
        self.k = k
        self.dataset = dataset
        self.names = names

    def get_nearest_neighbors(self, timeseries: list, dtw_type: str='vanilla', window_size: int=None) -> tuple:
        "Get the k-nearest neighbors for a given time series."
        assert len(timeseries) > 2, "The time series must have at least one element."
        assert dtw_type in ['vanilla', 'window'], "Invalid dtw_type. Choose between 'vanilla' and 'window'."
        if dtw_type == 'window':
            assert window_size is not None, "window_size must be provided for window DTW."
            assert window_size > 0, "window_size must be greater than 0."
        distances = [float('inf')] * self.k
        neighbors = [None] * self.k
        names_neighbors = [None] * self.k
        for i, ts in enumerate(self.dataset):
            if dtw_type == 'vanilla':
                dtw = VanillaDTW(timeseries, ts)
            elif dtw_type == 'window':
                dtw = WindowDTW(X=timeseries, Y=ts, window=window_size)
            else:
                raise ValueError("Invalid dtw_type. Choose between 'vanilla' and 'window'.")
            dtw.compute_cost_matrix()
            D = dtw.compute_accumulated_cost_matrix()
            if D[-1, -1] < max(distances):
                # add the new neighbor
                neighbors.append(ts)
                # remove the neighbor with the highest distance
                neighbors.pop(np.argmax(distances))
                # add the new neighbor name
                names_neighbors.append(self.names[i])
                # remove the neighbor name with the highest distance
                names_neighbors.pop(np.argmax(distances))
                # add the new distance
                distances.append(D[-1, -1])
                # remove the distance with the highest value
                distances.pop(np.argmax(distances))
        sorted_neighbors = [neighbors[i] for i in np.argsort(distances)]
        sorted_names_neighbors = [names_neighbors[i] for i in np.argsort(distances)]
        sorted_distances = [distances[i] for i in np.argsort(distances)]
        return sorted_neighbors, sorted_distances, sorted_names_neighbors


if __name__=='__main__':
    
    import time

    # Create two random time series cosine waves shifted by pi/2
    X_1 = np.random.normal(0, 1, 1000).tolist()
    X_2 = np.sin(np.linspace(0, 3*np.pi, 1000)).tolist() 
    X_3 = np.cos(np.linspace(np.pi, 3*np.pi + np.pi, 1000)).tolist()
    X_4 = np.cos(np.linspace(np.pi/2, 3*np.pi + np.pi/2, 1000)).tolist()
    X = [X_1, X_2, X_3, X_4]
    X_names = ['X_1', 'X_2', 'X_3', 'X_4']

    # time series to find neighbors for
    Y = np.sin(np.linspace(0, 3*np.pi, 1000))
    Y = Y.tolist()

    # Compute the k-nearest neighbors using vanilla DTW
    start_time = time.time()
    knn_dtw = KNN_DTW(X, k=4, names=X_names)
    neighbors, distances, names_neighbors = knn_dtw.get_nearest_neighbors(Y)
    print(f"--- {time.time() - start_time} seconds ---")
    print(distances)
    print(names_neighbors)

    # Compute the k-nearest neighbors using window DTW
    start_time = time.time()
    knn_dtw = KNN_DTW(X, k=4, names=X_names)
    neighbors, distances, names_neighbors = knn_dtw.get_nearest_neighbors(Y, dtw_type='window', window_size=500)
    print(f"--- {time.time() - start_time} seconds ---")
    print(distances)
    print(names_neighbors)