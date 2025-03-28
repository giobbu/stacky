import numpy as np
import scipy as sp
from numba import jit
import seaborn as sns

class VanillaDTW:
    """ Basic Dynamic Time Warping (DTW) implementation."""
    def __init__(self, X: list, Y: list, metric: str='euclidean'):
        assert metric == 'euclidean', 'Only the Euclidean metric is supported for now.'
        self.X = X
        self.Y = Y
        self.metric = metric

    def compute_cost_matrix(self) -> np.ndarray:
        " Compute the cost matrix using the specified metric."
        X, Y = np.atleast_2d(self.X, self.Y)
        self.C = sp.spatial.distance.cdist(X.T, Y.T, metric=self.metric)
        return self.C

    def compute_accumulated_cost_matrix(self) -> np.ndarray:
        " Compute the accumulated cost matrix using dynamic programming."
        assert hasattr(self, 'C'), 'Need to compute the cost matrix first'
        N = self.C.shape[0]
        M = self.C.shape[1]
        self.D = np.zeros((N, M))
        self.D[0, 0] = self.C[0, 0]
        for n in range(1, N):
            self.D[n, 0] = self.D[n-1, 0] + self.C[n, 0]
        for m in range(1, M):
            self.D[0, m] = self.D[0, m-1] + self.C[0, m]
        for n in range(1, N):
            for m in range(1, M):
                self.D[n, m] = self.C[n, m] + min(self.D[n-1, m], self.D[n, m-1], self.D[n-1, m-1])
        return self.D

    def compute_optimal_warping_path(self) -> np.ndarray:
        """Compute the warping path given an accumulated cost matrix
        """
        N = self.D.shape[0]
        M = self.D.shape[1]
        n = N - 1
        m = M - 1
        P = [(n, m)]
        while n > 0 or m > 0:
            if n == 0:
                cell = (0, m - 1)
            elif m == 0:
                cell = (n - 1, 0)
            else:
                val = min(self.D[n-1, m-1], self.D[n-1, m], self.D[n, m-1])
                if val == self.D[n-1, m-1]:
                    cell = (n-1, m-1)
                elif val == self.D[n-1, m]:
                    cell = (n-1, m)
                else:
                    cell = (n, m-1)
            P.append(cell)
            (n, m) = cell
        P.reverse()
        return np.array(P)

if __name__=='__main__':

    # Create two random time series cosine waves shifted by pi/2
    X = np.cos(np.linspace(0, 3*np.pi, 100))
    Y = np.cos(np.linspace(np.pi/2, 3*np.pi + np.pi/2, 100))
    # to list
    X = X.tolist()
    Y = Y.tolist()
    dtw = VanillaDTW(X, Y)
    C = dtw.compute_cost_matrix()
    D = dtw.compute_accumulated_cost_matrix()
    P = dtw.compute_optimal_warping_path()

    # Print the cost matrix, accumulated cost matrix, and optimal warping path
    print(' Cost matrix:')
    print(C)
    print(' Accumulated cost matrix:')
    print(D)
    print(' Optimal warping path:')
    print(P)

    import matplotlib.pyplot as plt

    # Plot the cost matrix and optimal warping path
    P = np.array(P) 
    fig = plt.figure(figsize=(9, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(C, cmap="YlGnBu", origin='lower', aspect='equal')
    plt.plot(P[:, 1], P[:, 0], marker='o', color='r', markersize=1)
    plt.clim([0, np.max(C)])
    plt.colorbar()
    plt.title('$C$ with optimal warping path')
    plt.xlabel('Sequence Y')
    plt.ylabel('Sequence X')
    fig.savefig('imgs/cost_matrix.png')

    # Plot the accumulated cost matrix and optimal warping path
    plt.subplot(1, 2, 2)
    plt.imshow(D, cmap="YlGnBu", origin='lower', aspect='equal')
    plt.plot(P[:, 1], P[:, 0], marker='o', color='r', markersize=1)
    plt.clim([0, np.max(D)])
    plt.colorbar()
    plt.title('$D$ with optimal warping path')
    plt.xlabel('Sequence Y')
    plt.ylabel('Sequence X')
    plt.tight_layout()
    fig.savefig('imgs/accumulated_cost_matrix.png')
    plt.show()

    # Plot the two time series and the optimal warping path
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_visible(False)
    # add grid
    ax.grid(True)
    for [map_x, map_y] in P:
        ax.plot([map_x, map_y], [X[map_x], Y[map_y]], '--k', linewidth=2, color='gray')
    ax.plot(X, '-ro', label='x', linewidth=4, markersize=8, markerfacecolor='black', markeredgecolor='salmon', alpha=0.5)
    ax.plot(Y, '-bo', label='y', linewidth=4, markersize=8, markerfacecolor='black', markeredgecolor='skyblue', alpha=0.5)
    ax.set_title("DTW Distance", fontsize=10, fontweight="bold")
    ax.legend(fontsize=20)
    plt.tight_layout()
    fig.savefig('imgs/dtw_distance.png')
    plt.show()

    