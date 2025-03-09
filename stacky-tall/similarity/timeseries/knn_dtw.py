from vanilla_dtw import VanillaDTW
import numpy as np

class KNN_DTW:
    "K-Nearest Neighbors using Dynamic Time Warping (DTW) as the similarity measure."
    def __init__(self, dataset, names, k=1):
        self.k = k
        self.dataset = dataset
        self.names = names

    def get_nearest_neighbors(self, timeseries):
        "Get the k-nearest neighbors for a given time series."
        distances = [float('inf')] * self.k
        neighbors = [None] * self.k
        names_neighbors = [None] * self.k
        for i, ts in enumerate(self.dataset):
            dtw = VanillaDTW(timeseries, ts)
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

    start_time = time.time()
    # Compute the k-nearest neighbors using DTW
    knn_dtw = KNN_DTW(X, k=4, names=X_names)
    neighbors, distances, names_neighbors = knn_dtw.get_nearest_neighbors(Y)
    print(f"--- {time.time() - start_time} seconds ---")
    print(distances)
    print(names_neighbors)