import numpy as np

def mmd_unbiased_with_pval(x, y, sigma, n_permutations=1000, seed=None):
    rng = np.random.default_rng(seed)
    n, d = x.shape
    m, d2 = y.shape
    assert d == d2

    # RBF kernel function
    def rbf_kernel(a, b):
        dists = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
        return np.exp(-dists**2 / (2 * sigma**2))

    # Original MMD
    k_xx = rbf_kernel(x, x)
    k_yy = rbf_kernel(y, y)
    k_xy = rbf_kernel(x, y)
    np.fill_diagonal(k_xx, 0)
    np.fill_diagonal(k_yy, 0)

    # mmd = 1/n^2 * sum_i sum_j k(x_i, x_j) + 1/m^2 * sum_i sum_j k(y_i, y_j) - 2/(nm) * sum_i sum_j k(x_i, y_j)
    mmd_obs = (k_xx.sum() / (n * (n - 1))
             + k_yy.sum() / (m * (m - 1))
             - 2 * k_xy.sum() / (n * m))

    # Permutation test
    z = np.vstack([x, y])
    total = n + m
    mmd_perms = []

    for _ in range(n_permutations):
        idx = rng.permutation(total)
        x_perm = z[idx[:n]]
        y_perm = z[idx[n:]]

        k_xx_perm = rbf_kernel(x_perm, x_perm)
        k_yy_perm = rbf_kernel(y_perm, y_perm)
        k_xy_perm = rbf_kernel(x_perm, y_perm)
        np.fill_diagonal(k_xx_perm, 0)
        np.fill_diagonal(k_yy_perm, 0)

        mmd_perm = (k_xx_perm.sum() / (n * (n - 1))
                  + k_yy_perm.sum() / (m * (m - 1))
                  - 2 * k_xy_perm.sum() / (n * m))
        mmd_perms.append(mmd_perm)

    # p-value: fraction of permutations with MMD >= observed MMD
    mmd_perms = np.array(mmd_perms)
    p_value = np.mean(mmd_perms >= mmd_obs)

    return mmd_obs, mmd_perms, p_value

if __name__ == "__main__":
    x_before = np.random.normal(loc=0, scale=1, size=(500, 2))
    x_after = np.random.normal(loc=0.5, scale=1.5, size=(500, 2))
    sigma = 1.0
    mmd, mmd_perms, pval = mmd_unbiased_with_pval(x_before, x_after, sigma, n_permutations=1000, seed=42)
    print(f"MMD: {mmd}, p-value: {pval}")

    import matplotlib.pyplot as plt
    plt.hist(mmd_perms, bins=30, alpha=0.7, label='MMD Permutations')
    plt.axvline(mmd, color='red', linestyle='dashed', linewidth=2, label='Observed MMD')
    plt.xlabel('MMD Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of MMD Permutations')
    plt.legend()
    plt.savefig('mmd_permutations.png')
    plt.show()

    # plot x_before and x_after
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(x_before[:, 0], x_before[:, 1], alpha=0.5, label='Before')
    plt.title('Before Treatment')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.subplot(1, 2, 2)
    plt.scatter(x_after[:, 0], x_after[:, 1], alpha=0.5, label='After', color='orange')
    plt.title('After Treatment')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.tight_layout()
    plt.savefig('covariate_drift.png')
    plt.show()