import numpy as np
import scipy as sp
from numba import jit

class SimpleDTW:
    """ Simple Dynamic Time Warping (DTW) implementation."""
    def __init__(self, X, Y, metric='euclidean'):
        self.X = X
        self.Y = Y
        self.metric = metric

    def compute_cost_matrix(self):
        " Compute the cost matrix using the specified metric."
        X, Y = np.atleast_2d(self.X, self.Y)
        self.C = sp.spatial.distance.cdist(X.T, Y.T, metric=self.metric)
        return self.C

    def compute_accumulated_cost_matrix(self):
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

    def compute_optimal_warping_path(self):
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
    X = [1, 2, 3, 4, 5]
    Y = [2, 3, 4, 10, 5]
    dtw = SimpleDTW(X, Y)
    C = dtw.compute_cost_matrix()
    D = dtw.compute_accumulated_cost_matrix()
    P = dtw.compute_optimal_warping_path()
    print(' Cost matrix:')
    print(C)
    print(' Accumulated cost matrix:')
    print(D)
    print(' Optimal warping path:')
    print(P)

    import matplotlib.pyplot as plt

    P = np.array(P) 
    plt.figure(figsize=(9, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(C, cmap='gray_r', origin='lower', aspect='equal')
    plt.plot(P[:, 1], P[:, 0], marker='o', color='r')
    plt.clim([0, np.max(C)])
    plt.colorbar()
    plt.title('$C$ with optimal warping path')
    plt.xlabel('Sequence Y')
    plt.ylabel('Sequence X')

    plt.subplot(1, 2, 2)
    plt.imshow(D, cmap='gray_r', origin='lower', aspect='equal')
    plt.plot(P[:, 1], P[:, 0], marker='o', color='r')
    plt.clim([0, np.max(D)])
    plt.colorbar()
    plt.title('$D$ with optimal warping path')
    plt.xlabel('Sequence Y')
    plt.ylabel('Sequence X')
    plt.tight_layout()
    plt.show()

    