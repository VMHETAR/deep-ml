import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(
    n_samples: int,
    k: int = 5,
    shuffle: bool = True
) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.
    """

    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    # Determine fold sizes
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1

    folds = []
    current = 0

    for fold_size in fold_sizes:
        folds.append(indices[current:current + fold_size].tolist())
        current += fold_size

    splits = []

    for i in range(k):
        test_indices = folds[i]

        train_indices = []
        for j in range(k):
            if j != i:
                train_indices.extend(folds[j])

        splits.append((train_indices, test_indices))

    return splits