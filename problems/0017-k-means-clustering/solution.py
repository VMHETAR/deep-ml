import numpy as np

def k_means_clustering(
    points: list[tuple[float, ...]],
    k: int,
    initial_centroids: list[tuple[float, ...]],
    max_iterations: int
) -> list[tuple[float, ...]]:

    points = np.array(points, dtype=float)
    centroids = np.array(initial_centroids, dtype=float)

    for _ in range(max_iterations):
        distances = []

        for point in points:
            point_distances = []

            for centroid in centroids:
                distance = np.linalg.norm(point - centroid)
                point_distances.append(distance)

            distances.append(point_distances)

        distances = np.array(distances)

        clusters = np.argmin(distances, axis=1)

        for cluster_id in range(k):
            cluster_points = points[clusters == cluster_id]
            new_centroid = np.mean(cluster_points, axis=0)

            centroids[cluster_id] = new_centroid

    return [tuple(centroid) for centroid in centroids]