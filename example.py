import numpy as np
from symmnmf import SymNMF
from matplotlib import pyplot as plt


np.random.seed(0)

A = np.array([
    [1, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0.5, 0.5, 0.5, 0.5], # partial membership
    [0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 1, 1, 1, 1],
])

A += np.abs(np.random.normal(scale=0.15, size=A.size)).reshape(A.shape)

snmf = SymNMF(k=3)
snmf.fit(A)

fig, axs = plt.subplots(1, 2, figsize=(7, 4))

axs[0].imshow(A)
axs[0].set_title("Affinity matrix")

axs[1].imshow(snmf.H)
axs[1].set_title("Cluster memberships")
axs[1].set_xlabel("cluster")
axs[1].set_ylabel("node")

fig.tight_layout()
fig.show()

