import numpy as np


def choose_main_element(A: np.ndarray) -> np.ndarray:
    n = A.shape[0]

    x_permutations = np.arange(0, n)
    for k in range(n):
        temp_matrix = A[k:n, k:n]
        i, j = np.unravel_index(np.abs(temp_matrix).argmax(), temp_matrix.shape)
        i += k
        j += k
        A[[k, i]] = A[[i, k]]
        A[:, [k, j]] = A[:, [j, k]]

        temp = x_permutations[k]
        x_permutations[k] = x_permutations[j]
        x_permutations[j] = temp

    return x_permutations


def gaussian_method(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    concat_A = np.append(A, b, axis=1)
    x_permutations = choose_main_element(concat_A)

    n = A.shape[0]
    # Forward
    for k in range(n):
        tmp = concat_A[k, k]
        if (tmp != 0):
            for j in range(k, n + 1):
                concat_A[k, j] /= tmp

        for i in range(k + 1, n):
            tmp = concat_A[i, k]
            for j in range(k, n + 1):
                concat_A[i, j] -= concat_A[k, j] * tmp

    # Backward
    X = np.empty([n, 1], dtype=np.float64)
    for i in reversed(range(n)):
        perm_index_i = x_permutations[i]
        X[perm_index_i] = concat_A[i, n]
        for j in range(i + 1, n):
            perm_index_j = x_permutations[j]
            X[perm_index_i] -= concat_A[i, j] * X[perm_index_j]

    return X


def make_simpson_coefs(a: float, b: float, n: int):
    h = (b - a) / n

    coefs = [0.0 for _ in range(n + 1)]
    points = [a + h * k for k in range(n + 1)]

    coefs[0] = h / 3
    for j in range(1, n // 2 - 1 + 1):
        coefs[2 * j] = h / 3 * 2.0

    for j in range(1, n // 2 + 1):
        coefs[2 * j - 1] = h / 3 * 4.0

    coefs[n] = h / 3

    return coefs, points


def kronecker_symbol(i: int, j: int):
    if i == j:
        return 1.0
    else:
        return 0.0
