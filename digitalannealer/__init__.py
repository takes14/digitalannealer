"""Digital Annealer: High-performance GPU QUBO/Ising solver.

The ``digitalannealer`` package provides GPU-accelerated samplers for
Ising and QUBO combinatorial optimisation problems.  The public API is
designed to be compatible with `dimod <https://docs.ocean.dwavesys.com/>`_.

Main entry points
-----------------
:class:`DASampler`
    Single-replica Digital Annealer sampler (dense and sparse modes).
:class:`PTDASampler`
    Parallel Tempering variant with multi-replica exchange.
:func:`qubo_to_ising` / :func:`ising_to_qubo`
    Conversion utilities between QUBO and Ising representations.

Example
-------
>>> from digitalannealer import DASampler
>>> sampler = DASampler(seed=42)
>>> result = sampler.sample_ising({0: -1.0}, {(0, 1): 0.5}, num_sweeps=100)
>>> print(result.first.energy)
"""

from digitalannealer.models.converters import ising_to_qubo, qubo_to_ising
from digitalannealer.samplers import DASampler, PTDASampler

__version__ = "0.1.0"
__all__ = ["DASampler", "PTDASampler", "qubo_to_ising", "ising_to_qubo"]
