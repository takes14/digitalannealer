Digital Annealer Documentation
===============================

**Digital Annealer** is a high-performance GPU QUBO/Ising solver with an
`OpenJij <https://github.com/OpenJij/OpenJij>`_ / `dimod <https://docs.ocean.dwavesys.com/en/stable/docs_dimod/>`_ compatible Python API
and a C++/CUDA backend.

All annealing computation — state updates and replica exchange — runs
entirely on the GPU to eliminate PCIe transfer overhead.

Features
--------

- **All-GPU Execution**: Annealing loop, parallel trial, and replica exchange
  all run inside a single CUDA kernel launch.
- **Massive Parallelism**: Parallel trial spin updates evaluate all candidate
  flips in a single sweep.
- **Dynamic Offset**: Rejection-based offset mechanism helps escape local
  minima.
- **Parallel Tempering**: Multiple replicas at different temperatures with
  GPU-resident replica exchange.
- **Dense & Sparse Modes**: Dense interaction matrix for fully-connected
  graphs; CSR + shared-memory path for sparse problems.
- **dimod Compatible**: Accepts ``dict`` / NumPy inputs and returns
  ``dimod.SampleSet``.

Getting Started
---------------

Install the package (requires CUDA toolkit and a compatible NVIDIA GPU):

.. code-block:: bash

   pip install git+https://github.com/takes14/digitalannealer.git

Quick example:

.. code-block:: python

   from digitalannealer import DASampler

   sampler = DASampler(seed=42)
   h = {0: -1.0, 1: 1.0}
   J = {(0, 1): -1.0}
   result = sampler.sample_ising(h, J, num_sweeps=1000)
   print(result.first.sample, result.first.energy)

   # For machine learning workflows that need the best sample as a NumPy `ndarray`
   min_index = np.argmin(result.record.energy)
   best_sample = result.record.sample[min_index]

.. toctree::
   :maxdepth: 2
   :caption: Contents

   getting-started/index
   reference/index
   architecture/index
   project/index
