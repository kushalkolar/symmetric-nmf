import numpy as np
from tqdm import tqdm

from .nnls import nnlsm_blockpivot


class SymNMF:
    def __init__(
            self,
            A: np.ndarray,
            k: int,
            H: np.ndarray = None,
            maxiter: int = 10_000,
            tol: float = 10e-3,
            alpha: float = None,
    ):
        self.A = A

        if H is None:
            self.H = 2 * np.sqrt(np.mean(np.mean(A) / k)) * np.random.rand(self.A.shape[0], k)

        else:
            self.H = H

        if self.H.shape[0] != A.shape[0]:
            raise ValueError

        if alpha is None:
            self.alpha = np.max(A) ** 2
        else:
            self.alpha = alpha

        self.maxiter = maxiter
        self.tol = tol
        self.k = k

    def fit(self):
        self.W = self.H
        self.I_k = self.alpha * np.eye(self.k)

        self.left = self.H.conj().T @ self.H
        self.right = self.A @ self.H

        for self.i in tqdm(range(self.maxiter)):
            self.W = nnlsm_blockpivot(
                self.left + self.I_k,
                (self.right + self.alpha * self.H).conj().T,
                True,
                self.W.conj().T
            )[0].T

            self.left = self.W.conj().T @ self.W
            self.right = self.A @ self.W

            self.H = nnlsm_blockpivot(
                self.left + self.I_k,
                (self.right + self.alpha * self.W).conj().T,
                True,
                self.H.conj().T
            )[0].T
            # tempW = W.sum(axis=1)
            # tempH = H.sum(axis=1)

            temp = self.alpha * (self.H - self.W)

            gradH = self.H @ self.left - self.right + temp
            self.left = self.H.conj().T @ self.H
            self.right = self.A @ self.H

            gradW = self.W @ self.left - self.right - temp

            W_norm = np.linalg.norm(gradW[(gradW <= 0) | (self.W > 0)])
            H_norm = np.linalg.norm(gradH[(gradH <= 0) | (self.H > 0)])

            if self.i == 0:
                initgrad = np.sqrt(W_norm ** 2 + H_norm ** 2)
            else:
                projnorm = np.sqrt(W_norm ** 2 + H_norm ** 2)
                if projnorm < self.tol * initgrad:
                    break
