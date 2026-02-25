from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.cluster import KMeans


def sample_kmeans(
    X: np.ndarray, y: np.ndarray, n_samples: int, random_state: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample using K-means clustering centroids.

    Returns:
        centroid_X: Cluster centroids (features)
        centroid_y: Labels from closest samples to centroids
        centroid_indices: Indices of closest samples to centroids
    """
    kmeans = KMeans(n_clusters=n_samples, random_state=random_state, n_init=10)
    kmeans.fit(X.T)
    centroids = kmeans.cluster_centers_.T
    distances = np.linalg.norm(X.T[:, np.newaxis, :] - kmeans.cluster_centers_, axis=2)
    centroid_indices = np.argmin(distances, axis=0)
    centroid_y = y[centroid_indices]
    return centroids, centroid_y, centroid_indices


def sample_kmedoids(
    X: np.ndarray, y: np.ndarray, n_samples: int, random_state: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample using K-medoids clustering (uses actual data points).

    Returns:
        medoid_X: Medoid samples (features)
        medoid_y: Labels of medoid samples
        medoid_indices: Indices of medoid samples
    """
    try:
        from sklearn_extra.cluster import KMedoids

        kmedoids = KMedoids(n_clusters=n_samples, random_state=random_state, method="pam")
        kmedoids.fit(X.T)
        medoid_indices = kmedoids.medoid_indices_
        medoid_X = X[:, medoid_indices]
        medoid_y = y[medoid_indices]
        return medoid_X, medoid_y, medoid_indices
    except ImportError:
        # Fallback: use K-means if sklearn-extra not available
        print("Warning: sklearn-extra not available, using K-means for kmedoids")
        return sample_kmeans(X, y, n_samples, random_state)


def sample_farthest_point(
    X: np.ndarray, y: np.ndarray, n_samples: int, random_state: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample using Farthest Point Sampling (FPS).
    Iteratively selects points farthest from already selected points.

    Returns:
        sample_X: Selected samples (features)
        sample_y: Labels of selected samples
        sample_indices: Indices of selected samples
    """
    np.random.seed(random_state)
    n_total = X.shape[1]
    if n_samples >= n_total:
        return X, y, np.arange(n_total)

    selected_indices = []
    # Start with a random point
    selected_indices.append(np.random.randint(0, n_total))

    # Iteratively select farthest points
    for _ in range(n_samples - 1):
        selected_points = X[:, selected_indices].T  # Shape: (n_selected, n_features)
        distances = np.linalg.norm(
            X.T[:, np.newaxis, :] - selected_points, axis=2
        )  # Shape: (n_total, n_selected)
        min_distances = np.min(distances, axis=1)  # Minimum distance to any selected point
        # Select point with maximum minimum distance
        farthest_idx = np.argmax(min_distances)
        selected_indices.append(farthest_idx)

    selected_indices = np.array(selected_indices)
    return X[:, selected_indices], y[selected_indices], selected_indices


def sample_stratified(
    X: np.ndarray, y: np.ndarray, n_samples: int, random_state: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample using stratified sampling to maintain class distribution.

    Returns:
        sample_X: Selected samples (features)
        sample_y: Labels of selected samples
        sample_indices: Indices of selected samples
    """
    np.random.seed(random_state)
    n_total = X.shape[1]
    if n_samples >= n_total:
        return X, y, np.arange(n_total)

    unique_classes, class_counts = np.unique(y, return_counts=True)
    n_classes = len(unique_classes)

    selected_indices = []
    samples_per_class = n_samples // n_classes
    remainder = n_samples % n_classes

    for i, class_label in enumerate(unique_classes):
        class_indices = np.where(y == class_label)[0]
        n_samples_this_class = samples_per_class + (1 if i < remainder else 0)
        n_samples_this_class = min(n_samples_this_class, len(class_indices))

        if n_samples_this_class > 0:
            selected = np.random.choice(class_indices, size=n_samples_this_class, replace=False)
            selected_indices.extend(selected)

    selected_indices = np.array(selected_indices)
    return X[:, selected_indices], y[selected_indices], selected_indices


def sample_random(
    X: np.ndarray, y: np.ndarray, n_samples: int, random_state: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample using random selection.

    Returns:
        sample_X: Selected samples (features)
        sample_y: Labels of selected samples
        sample_indices: Indices of selected samples
    """
    np.random.seed(random_state)
    n_total = X.shape[1]
    if n_samples >= n_total:
        return X, y, np.arange(n_total)

    selected_indices = np.random.choice(n_total, size=n_samples, replace=False)
    return X[:, selected_indices], y[selected_indices], selected_indices


def get_sampling_function(method: str):
    """Get the sampling function for the given method name."""
    sampling_functions = {
        "kmeans": sample_kmeans,
        "kmedoids": sample_kmedoids,
        "farthest_point": sample_farthest_point,
        "stratified": sample_stratified,
        "random": sample_random,
    }
    if method not in sampling_functions:
        raise ValueError(
            f"Unknown sampling method: {method}. Available: {list(sampling_functions.keys())}"
        )
    return sampling_functions[method]


