import numpy as np
import scipy as sp
from numba import jit
import seaborn as sns

class WindowDTW:
    """ Basic Dynamic Time Warping (DTW) implementation."""
    def __init__(self, X: list, Y: list, window: int, metric: str='euclidean'):
        assert metric == 'euclidean', 'Only the Euclidean metric is supported for now.'
        self.X = X
        self.Y = Y
        self.metric = metric
        self.window = self._enforce_locality_constraint(window)

    def _enforce_locality_constraint(self, window: int) -> int:
        " Compute the window size."
        assert window >= 0, 'The window size must be greater than or equal to 0.'
        return max(window, abs(len(self.X)-len(self.Y)))

    def compute_cost_matrix(self) -> np.ndarray:
        " Compute the cost matrix using the specified metric."
        X, Y = np.array(self.X), np.array(self.Y)
        X, Y = np.atleast_2d(X, Y)
        self.C = sp.spatial.distance.cdist(X.T, Y.T, metric=self.metric)  # Compute the pairwise distances
        return self.C

    def compute_accumulated_cost_matrix(self) -> np.ndarray:
        " Compute the accumulated cost matrix using dynamic programming."
        assert hasattr(self, 'C'), 'Need to compute the cost matrix first'
        N, M = self.C.shape[0], self.C.shape[1]
        self.D = np.zeros((N, M))
        self.D[0, 0] = self.C[0, 0]
        # Initialize the first row and column
        for n in range(1, N):
            self.D[n, 0] = self.D[n-1, 0] + self.C[n, 0]
        for m in range(1, M):
            self.D[0, m] = self.D[0, m-1] + self.C[0, m]
        # Fill in the rest of the matrix
        for n in range(1, N):
            if self.window == 0:  # Classic DTW
                for m in range(1, M):
                    self.D[n, m] = self.C[n, m] + min(self.D[n-1, m], self.D[n, m-1], self.D[n-1, m-1])
            else:  # DTW with a window
                for m in range(max(1, n-self.window), min(M, n+self.window)):
                    self.D[n, m] = self.C[n, m] + min(self.D[n-1, m], self.D[n, m-1], self.D[n-1, m-1])
        return self.D

if __name__=='__main__':

    import time

    # Create two random time series cosine waves shifted by pi/2
    X = np.cos(np.linspace(0, 3*np.pi, 1000))
    Y = np.cos(np.linspace(np.pi/2, 3*np.pi + np.pi/2, 1000))
    # to list to make it work with the code
    X = X.tolist()
    Y = Y.tolist()

    start_time = time.time()
    window = 10
    # Compute the cost matrix and accumulated cost matrix using windowed DTW window=10
    windowed_dtw = WindowDTW(X, Y, window=window)
    C = windowed_dtw.compute_cost_matrix()
    D = windowed_dtw.compute_accumulated_cost_matrix()
    print(f'Similarity: {D[-1, -1]} for window size: {window}')
    print(' Time taken: ', time.time() - start_time)

    start_time = time.time()
    window = 100
    # Compute the cost matrix and accumulated cost matrix using windowed DTW window=100
    windowed_dtw = WindowDTW(X, Y, window=window)
    C = windowed_dtw.compute_cost_matrix()
    D = windowed_dtw.compute_accumulated_cost_matrix()
    print(f'Similarity: {D[-1, -1]} for window size: {window}')
    print(' Time taken: ', time.time() - start_time)

    start_time = time.time()
    window = 250
    # Compute the cost matrix and accumulated cost matrix using windowed DTW window=250
    windowed_dtw = WindowDTW(X, Y, window=window)
    C = windowed_dtw.compute_cost_matrix()
    D = windowed_dtw.compute_accumulated_cost_matrix()
    print(f'Similarity: {D[-1, -1]} for window size: {window}')
    print(' Time taken: ', time.time() - start_time)

    start_time = time.time()
    window = 500
    # Compute the cost matrix and accumulated cost matrix using windowed DTW window=500
    windowed_dtw = WindowDTW(X, Y, window=window)
    C = windowed_dtw.compute_cost_matrix()
    D = windowed_dtw.compute_accumulated_cost_matrix()
    print(f'Similarity: {D[-1, -1]} for window size: {window}')
    print(' Time taken: ', time.time() - start_time)

    start_time = time.time()
    # Compute the cost matrix and accumulated cost matrix using basic DTW
    basic_dtw = WindowDTW(X, Y, window=0)
    C = basic_dtw.compute_cost_matrix()
    D = basic_dtw.compute_accumulated_cost_matrix()
    print('Similarity:', D[-1, -1])
    print(' Time taken: ', time.time() - start_time)

