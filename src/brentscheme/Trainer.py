"""Multi-step training utilities for Brent schemes."""

from __future__ import annotations

import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm, trange


device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(slots=True)
class Trainer:
    """Coordinate repeated optimization steps on a scheme.

    Notes
    -----
    This class intentionally remains API-compatible with the original interface.
    It primarily tracks epoch counts and delegates per-step logic to :class:`Stepper`.
    """

    num_epochs: int = 0

    def train(
        self,
        scheme,
        epochs: int = 200,
        batch_size: int = 1,
        lr: float = 1e-7,
        momentum: float = 0.9,
        use_L2: bool = False,
        penalty: float = 0.0,
        verbose: int = 0,
    ) -> None:
        """Train a scheme in-place.

        Parameters are retained for backward compatibility with existing notebooks.
        """
        from brentscheme.SchemeDisplay import SchemeDisplay
        from brentscheme.Stepper import Stepper

        stepper = Stepper()
        display = SchemeDisplay()
        y = [list(display.metrics(scheme).as_tuple())]

        start = time.perf_counter()
        for _ in range(epochs):
            if use_L2:
                stepper.epoch_pseudoinverse(scheme, batch_size=batch_size, verbose=0)
            else:
                stepper.epoch(
                    scheme,
                    batch_size=batch_size,
                    lr=lr,
                    momentum=momentum,
                    penalty=penalty,
                )
            y.append(list(display.metrics(scheme).as_tuple()))

        self.num_epochs += epochs * batch_size
        runtime = time.perf_counter() - start

        if verbose > 1:
            plt.plot([i for i in range(epochs + 1)], [j[0] for j in y], color="blue", label="L1")
            plt.plot([i for i in range(epochs + 1)], [j[1] for j in y], color="black", label="L2")
            plt.plot([i for i in range(epochs + 1)], [j[2] for j in y], color="red", label="Linf")
            plt.title(f"Normalized Error For n={scheme.n}, p={scheme.p}: Ran in {runtime:.4f} sec.")
            plt.xlabel(f"Number of Epochs: {self.num_epochs}")
            plt.ylabel("Average Error of Output Entries (Log 10)")
            plt.grid(axis="y")
            plt.legend()
            plt.show()

    def optimize_basis(self, scheme, batch_size=1000, lr=1e-6, loss_norm=np.inf, verbose=0):
        """Optimize a basis change to improve non-L2 error for a fixed scheme."""
        from brentscheme.SchemeDisplay import SchemeDisplay
        from brentscheme.SchemeManipulator import SchemeManipulator
        from brentscheme.utils.tensors import random_unitary

        printer = SchemeDisplay()
        manipulator = SchemeManipulator()

        loss_fn = nn.L1Loss()
        pos = 0
        if loss_norm == np.inf:
            loss_fn = lambda x, y: torch.max(torch.abs(x - y))
            pos = 2

        score1 = printer.error(scheme)
        score2 = [np.inf] * 3

        while score2[pos] >= score1:
            L = torch.nn.Parameter(torch.eye(scheme.n) + 1e-14 * random_unitary(scheme.n)).type(torch.float64).to(device)
            M = torch.nn.Parameter(torch.eye(scheme.d) + 1e-14 * random_unitary(scheme.d)).type(torch.float64).to(device)
            R = torch.nn.Parameter(torch.eye(scheme.m) + 1e-14 * random_unitary(scheme.m)).type(torch.float64).to(device)
            test_scheme = scheme.clone()
            manipulator.change_basis(test_scheme, L=L, M=M, R=R)
            score2 = printer.error(test_scheme)

        optimizer = optim.Adam([L, M, R], lr=lr)

        for i in range(batch_size):
            test_scheme = scheme.clone()
            optimizer.zero_grad()
            manipulator.change_basis(test_scheme, L=L, M=M, R=R)
            output = test_scheme.forward()
            target = test_scheme.TRIPLE_DELTA_nmnddm
            cost = loss_fn(output, target)
            cost.backward()
            optimizer.step()
            if verbose > 0 and i % max(1, (batch_size // 10)) == 0:
                print(cost.item())

        if cost < 10**score1[pos]:
            return L.cpu().detach().type(torch.float64), M.cpu().detach().type(torch.float64), R.cpu().detach().type(torch.float64)
        return torch.eye(scheme.n), torch.eye(scheme.d), torch.eye(scheme.m)
