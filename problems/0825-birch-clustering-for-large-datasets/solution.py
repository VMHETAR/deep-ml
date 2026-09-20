import numpy as np


def birch_cluster(X, threshold):

    clusters = []

    for x in X:
        x = np.asarray(x, dtype=float)

        # 1. No subclusters yet
        if not clusters:
            clusters.append({
                "N": 1,
                "LS": x.copy(),
                "SS": x ** 2
            })
            continue

        # 2. Find closest centroid
        distances = []

        for cf in clusters:
            centroid = cf["LS"] / cf["N"]
            distance = np.linalg.norm(x - centroid)
            distances.append(distance)

        # np.argmin returns the first occurrence,
        # which gives the required smaller-index tie break.
        closest_idx = np.argmin(distances)
        cf = clusters[closest_idx]

        # 3. Tentatively absorb x
        new_N = cf["N"] + 1
        new_LS = cf["LS"] + x
        new_SS = cf["SS"] + x ** 2

        # New radius
        variance = (
            new_SS / new_N
            - (new_LS / new_N) ** 2
        )

        # Clamp floating-point noise
        variance = np.maximum(variance, 0.0)

        radius = np.sqrt(np.sum(variance))

        # 4. Accept or reject absorption
        if radius <= threshold:
            cf["N"] = new_N
            cf["LS"] = new_LS
            cf["SS"] = new_SS

        else:
            # Create a new subcluster
            clusters.append({
                "N": 1,
                "LS": x.copy(),
                "SS": x ** 2
            })

    # 5. Extract centroids
    centroids = [
        (cf["LS"] / cf["N"]).tolist()
        for cf in clusters
    ]

    # 6. Lexicographic sorting
    centroids.sort()

    return centroids