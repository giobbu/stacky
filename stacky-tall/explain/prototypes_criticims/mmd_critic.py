import numpy as np
import matplotlib.pyplot as plt

import numpy as np

def rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    """
    Computes the RBF kernel between two sets of points.
    """
    assert X.ndim == 2 and Y.ndim == 2, "Both X and Y should be 2D arrays."
    assert X.shape[1] == Y.shape[1], "X and Y should have the same number of features."
    assert gamma > 0, "Gamma should be a positive value."
    assert X.shape[0] > 0 and Y.shape[0] > 0, "X and Y should have at least one sample."
    X_norm = np.sum(X**2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y**2, axis=1).reshape(1, -1)
    K = np.exp(-gamma * (X_norm + Y_norm - 2 * np.dot(X, Y.T)))
    return K


def greedy_sel_prot(X: np.ndarray, num_prot: int, gamma: float) -> tuple:
    """
    Greedy selection of prototypes using MMD².
    """
    assert num_prot > 0, "Number of prototypes should be positive."
    n = len(X)
    K_XX = rbf_kernel(X, X, gamma)
    score_xx = np.sum(K_XX)/(n * n)
    selected_mask = np.zeros(n, dtype=bool)
    selected_indices = []
    mmd2_final = 0
    for _ in range(num_prot):
        m = len(selected_indices)
        if m > 0:
            # Get previously selected columns
            k_zz = K_XX[:, selected_indices]
            # Calculate scores only for unselected points
            unselected = ~selected_mask
            score_zz = np.sum(k_zz[unselected], axis=1) / (m * m)
            score_zx = np.sum(K_XX[unselected], axis=1) / (n * m)
            mmd2 = -2 * score_zx + score_zz + score_xx
        else:
            # First selection: all points have the same score (score_xx)
            unselected = np.ones(n, dtype=bool)
            mmd2 = np.ones(n) * score_xx
        # Find best candidate among unselected points
        best_idx = np.where(unselected)[0][np.argmin(mmd2)]
        # Update selected points
        selected_indices.append(best_idx)
        selected_mask[best_idx] = True
        mmd2_final += mmd2[best_idx]
    return selected_indices, mmd2_final

def sel_crit(X_without_Z: np.ndarray, Z: np.ndarray, num_crit: int, gamma: float) -> tuple:
    """
    Select the best criticisms based on witness.
    """
    assert num_crit > 0, "Number of criticisms should be positive."
    K_XX = rbf_kernel(X_without_Z, X_without_Z, gamma)
    K_XZ = rbf_kernel(X_without_Z, Z, gamma)
    witness_scores = np.mean(K_XX, axis=1)  - np.mean(K_XZ, axis=1)
    sorted_indices = np.argsort(witness_scores)[::-1]
    selected_indices = sorted_indices[:num_crit]
    values_witness = witness_scores[selected_indices]
    return selected_indices, values_witness


if __name__ == "__main__":
    # Example usage
    X1 = np.random.normal(100, 5, (100, 2))
    X2 = np.random.normal(5, 5, (100, 2))
    X3 = np.random.normal(50, 5, (100, 2))
    X = np.vstack((X1, X2, X3))
    # add anomalies
    num_anomalies = 10
    anomalies = np.random.uniform(0, 100, (num_anomalies, 2))
    X = np.vstack((X, anomalies))


    num_prot = 3
    cand_idx = np.arange(len(X))
    gamma = .1

    selected_prot, mmd_score = greedy_sel_prot(X, num_prot, gamma)
    print("Selected prototypes:", selected_prot)

    X_without_Z = np.delete(X, selected_prot, axis=0)
    num_crit = 10
    selected_crit, values_witness = sel_crit(X_without_Z, X[selected_prot], num_crit, gamma)
    print("Selected criticisms:", selected_crit)

    # Plot in scatter
    plt.scatter(X[:, 0], X[:, 1], c='blue', label='Data Points')
    for selected in selected_prot:
        plt.scatter(X[selected, 0], X[selected, 1], label=f'Prototype {selected}', color='red')
    for i, selected in enumerate(selected_crit):
        plt.scatter(X[selected, 0], X[selected, 1], 
                    label=f'Criticism {selected} with score {values_witness[i]:.2f}', 
                    color='green', marker='x')
        
    plt.title(f'Greedy Search of Prototypes \nMMD² Score: {mmd_score:.4f}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()



