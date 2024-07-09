import numpy as np


class SymNMF:
    def __init__(
            self,
            A: np.ndarray,
            k: int,
            H: np.ndarray = None,
            maxiter: int = 10_000,
            tol: float = 10e-4,
            sigma: float = 0.1,
            beta: float = 0.1,
    ):
        n = A.shape[0]

        if H is None:
            H = 2 * np.sqrt(np.mean(np.mean(A) / k) * np.random.rand(n, k)

