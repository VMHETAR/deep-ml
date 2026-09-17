import numpy as np


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least squares with an intercept.

    Args:
        X (np.ndarray): (n, p) design matrix without an intercept column.
        y (np.ndarray): (n,) target.

    Returns:
        np.ndarray: (p + 1,) coefficients, intercept first.
    """
    X_design = np.column_stack([np.ones(len(X)), X])

    beta, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)

    return beta


def omitted_variable_bias(X: np.ndarray, y: np.ndarray, omit_idx: int) -> tuple:
    """Return (full_kept, short, bias) for a two-column X."""

    # Which variable are we keeping?
    kept_idx = 1 - omit_idx

    # 1. Full regression: y ~ x1 + x2
    full = ols(X, y)

    # Coefficient of the kept variable
    full_kept = full[kept_idx + 1]

    # 2. Short regression: y ~ kept variable
    X_short = X[:, [kept_idx]]
    short_fit = ols(X_short, y)

    short = short_fit[1]

    # 3. Auxiliary regression:
    # omitted variable ~ kept variable
    omitted = X[:, omit_idx]
    kept = X[:, [kept_idx]]

    auxiliary = ols(kept, omitted)
    delta = auxiliary[1]

    # Omitted variable's coefficient from the full model
    beta_omitted = full[omit_idx + 1]

    # Omitted-variable bias
    bias = beta_omitted * delta

    return full_kept, short, bias