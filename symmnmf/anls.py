import numpy as np
from tqdm import tqdm

from .nnls import nnlsm_blockpivot


class SymNMF:
    """Symmetric NMF using ANLS"""
    def __init__(
            self,
            k: int,
            maxiter: int = 10_000,
            tol: float = 10e-3,
            alpha: float = None,
    ):

        self.alpha_ = alpha
        self.maxiter_ = maxiter
        self.tol_ = tol
        self.k_ = k

    def fit(self, A: np.ndarray, H_init: np.ndarray = None):
        self.A = A

        if H_init is None:
            H_init = 2 * np.sqrt(np.mean(np.mean(A) / self.k_)) * np.random.rand(self.A.shape[0], self.k_)
        else:
            if H_init.shape[0] != A.shape[0]:
                raise ValueError(f"H_init.shape[0] != A.shape[0], {H_init.shape[0]} != {A.shape[0]}")

        self.H_init_ = H_init

        if self.alpha_ is None:
            self.alpha_ = np.max(self.A) **2

        self.H = self.H_init_.copy()

        self.W = self.H
        I_k = self.alpha_ * np.eye(self.k_)

        self.left = self.H.conj().T @ self.H
        self.right = self.A @ self.H

        for i in tqdm(range(self.maxiter_)):
            self.W = nnlsm_blockpivot(
                self.left + I_k,
                (self.right + self.alpha_ * self.H).conj().T,
                True,
                self.W.conj().T
            )[0].T

            self.left = self.W.conj().T @ self.W
            self.right = self.A @ self.W

            self.H = nnlsm_blockpivot(
                self.left + I_k,
                (self.right + self.alpha_ * self.W).conj().T,
                True,
                self.H.conj().T
            )[0].T
            # tempW = W.sum(axis=1)
            # tempH = H.sum(axis=1)

            temp = self.alpha_ * (self.H - self.W)

            gradH = self.H @ self.left - self.right + temp
            self.left = self.H.conj().T @ self.H
            self.right = self.A @ self.H

            gradW = self.W @ self.left - self.right - temp

            W_norm = np.linalg.norm(gradW[(gradW <= 0) | (self.W > 0)])
            H_norm = np.linalg.norm(gradH[(gradH <= 0) | (self.H > 0)])

            if i == 0:
                initgrad = np.sqrt(W_norm ** 2 + H_norm ** 2)
            else:
                projnorm = np.sqrt(W_norm ** 2 + H_norm ** 2)
                if projnorm < self.tol_ * initgrad:
                    break
