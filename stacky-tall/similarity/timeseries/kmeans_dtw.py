import random
import numpy as np
from window_dtw import WindowDTW
from tslearn.barycenters import euclidean_barycenter
from loguru import logger
import matplotlib.pyplot as plt

def k_means_clust(dataset, num_clusters, num_iterations, window=5, tol=1e-4):
    """
    Perform K-means clustering using DTW distance as the similarity measure.
    """
    centroids = random.choices(dataset, k=num_clusters)  # Initialize centroids randomly
    for iteration in range(num_iterations):
        logger.info(f"Iteration {iteration+1}: Performing E-step")
        
        cluster_assignments = {i: [] for i in range(num_clusters)}
        total_inertia = 0  # Accumulate inertia

        for idx, ts in enumerate(dataset):
            min_distance = float('inf')
            closest_cluster = 0  # Default to first cluster
            for cluster_idx, ts_centroid in enumerate(centroids):
                dtw = WindowDTW(ts, ts_centroid, window)
                dtw.compute_cost_matrix()
                distances = dtw.compute_accumulated_cost_matrix()
                current_distance = distances[-1, -1]
                if current_distance < min_distance:
                    min_distance = current_distance
                    closest_cluster = cluster_idx
            cluster_assignments[closest_cluster].append(idx)

        logger.info("Performing M-step")
        for cluster_idx, cluster_indices in cluster_assignments.items():
            cluster_timeseries = [dataset[i] for i in cluster_indices]
            if cluster_timeseries:
                baricenter = euclidean_barycenter(cluster_timeseries).reshape(-1)
                centroids[cluster_idx] = baricenter
        
    return centroids, cluster_assignments

if __name__ == '__main__':
    num_clusters = 3
    num_timeseries = 5
    num_iterations = 20
    seq_length = 100

    dataset = []
    for _ in range(num_timeseries):
        dataset.append([np.sin(2 * np.pi * t / seq_length + random.uniform(0, 1)) for t in range(seq_length)])
    for _ in range(num_timeseries):
        dataset.append([np.cos(2 * np.pi * t / seq_length + random.uniform(0, 1)) for t in range(seq_length)])
    for _ in range(num_timeseries):
        dataset.append(np.cumsum(np.random.randn(seq_length)).tolist())  # Random walk

    centroids, cluster_assignments = k_means_clust(dataset, num_clusters, num_iterations, window=seq_length)

    # Plot clustered time series
    fig, axs = plt.subplots(num_clusters, sharex=True, sharey=True, figsize=(8, 6))
    for cluster_idx, indices in cluster_assignments.items():
        for i in indices:
            axs[cluster_idx].plot(dataset[i], alpha=0.5, color=f'C{cluster_idx}')
            # plot centroid
            axs[cluster_idx].plot(centroids[cluster_idx], color=f'C{cluster_idx}', linestyle='--')
        axs[cluster_idx].set_title(f'Cluster {cluster_idx} with num timeseries: {len(indices)}')
    plt.suptitle(f'Time series clustered using K-means with DTW: number of iterations: {num_iterations}')
    plt.show()