def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    """
    Transpose a 2D matrix by swapping rows and columns.
    
    Args:
        a: A 2D matrix of shape (m, n)
    
    Returns:
        The transposed matrix of shape (n, m)
    """
    # Your code here
    
    if len(a) is 0:
        return []
    result = []
    for i in range(len(a[0])):
        row = []
        for j in range(len(a)):
            row.append(a[j][i])
        result.append(row)
    return result