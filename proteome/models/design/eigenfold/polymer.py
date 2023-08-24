import numpy as np

from proteome.models.design.eigenfold.sde import HarmonicSDE


class PolymerSDE(HarmonicSDE):
    def __init__(self, N, a, b):
        super(PolymerSDE, self).__init__(
            N=N,
            edges=zip(np.arange(N - 1), np.arange(1, N)),
            antiedges=zip(np.arange(N - 2), np.arange(2, N)),
            a=a,
            b=b,
        )
