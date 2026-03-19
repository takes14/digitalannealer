Tutorial
========

This tutorial walks through the basic usage of the Digital Annealer library.

Prerequisites
-------------

- Python 3.10+
- NVIDIA GPU with CUDA support (CPU fallback is available for development)
- ``digitalannealer`` installed (see :doc:`/index` for installation)

Ising Model Basics
------------------

The Ising model defines an energy function over binary spin variables
:math:`s_i \in \{-1, +1\}`:

.. math::

   E = \sum_i h_i s_i + \sum_{i<j} J_{ij} s_i s_j

where :math:`h_i` are linear biases and :math:`J_{ij}` are quadratic
couplings.  The goal is to find the spin configuration that **minimises**
the energy.

Solving an Ising Problem with DASampler
---------------------------------------

The simplest way to use the library is via :class:`~digitalannealer.samplers.da_sampler.DASampler`:

.. code-block:: python

   from digitalannealer import DASampler

   # Define a small Ising problem
   h = {0: -1.0, 1: 0.5, 2: 0.0}
   J = {(0, 1): -1.0, (1, 2): 0.8}

   sampler = DASampler(seed=42)
   result = sampler.sample_ising(
       h, J,
       num_sweeps=1000,
       num_reads=10,
       beta_min=0.1,
       beta_max=10.0,
   )

   # Best solution found
   print("Best sample:", result.first.sample)
   print("Best energy:", result.first.energy)

``num_reads`` controls how many independent annealing runs to perform.
Each run starts from a random initial spin configuration.

Using Parallel Tempering (PTDASampler)
--------------------------------------

For harder problems with many local minima, Parallel Tempering runs
multiple replicas at different temperatures and exchanges configurations:

.. code-block:: python

   from digitalannealer import PTDASampler

   h = {0: -1.0, 1: 0.5, 2: 0.0, 3: -0.5}
   J = {(0, 1): -1.0, (1, 2): 0.8, (2, 3): -0.6, (0, 3): 0.4}

   sampler = PTDASampler(seed=42)
   result = sampler.sample_ising(
       h, J,
       num_sweeps=2000,
       num_reads=5,
       num_replicas=8,
       beta_min=0.1,
       beta_max=5.0,
       exchange_interval=10,
   )

   print("Best energy:", result.first.energy)

``num_replicas`` is the number of temperature replicas.
``exchange_interval`` controls how often adjacent replicas attempt
configuration swaps.

Solving a QUBO Problem
----------------------

QUBO (Quadratic Unconstrained Binary Optimization) problems use binary
variables :math:`x_i \in \{0, 1\}`:

.. math::

   E = \sum_{i \le j} Q_{ij} x_i x_j

Use :meth:`~digitalannealer.samplers.da_sampler.DASampler.sample_qubo` to solve
QUBO problems directly — the library automatically converts to Ising
internally:

.. code-block:: python

   from digitalannealer import DASampler

   # Max-Cut on a triangle graph
   Q = {
       (0, 0): -2.0, (1, 1): -2.0, (2, 2): -2.0,
       (0, 1): 2.0, (1, 2): 2.0, (0, 2): 2.0,
   }

   sampler = DASampler(seed=42)
   result = sampler.sample_qubo(Q, num_sweeps=1000, num_reads=10)

   print("Best sample:", result.first.sample)  # Binary {0, 1} values
   print("Best energy:", result.first.energy)

Converting Between QUBO and Ising
----------------------------------

You can also convert between representations explicitly:

.. code-block:: python

   from digitalannealer import qubo_to_ising, ising_to_qubo
   import numpy as np

   # QUBO -> Ising
   Q = {(0, 0): -1, (1, 1): -1, (0, 1): 2}
   h, J, offset = qubo_to_ising(Q)
   print("h:", h)
   print("J:", J)
   print("offset:", offset)

   # Ising -> QUBO
   h = np.array([-1.0, 1.0])
   J = np.array([[0.0, -0.5], [-0.5, 0.0]])
   Q_matrix, offset = ising_to_qubo(h, J)
   print("Q:", Q_matrix)

Using NumPy Arrays
------------------

Both samplers accept ``dict`` or NumPy array inputs:

.. code-block:: python

   import numpy as np
   from digitalannealer import DASampler

   n = 50
   rng = np.random.default_rng(42)
   h = rng.standard_normal(n)
   J = np.zeros((n, n))
   for i in range(n):
       for j in range(i + 1, n):
           if rng.random() < 0.3:
               val = rng.standard_normal()
               J[i, j] = val
               J[j, i] = val

   sampler = DASampler(seed=0)
   result = sampler.sample_ising(h, J, num_sweeps=2000, num_reads=5)
   print("Best energy:", result.first.energy)

   # For machine learning workflows that need the best sample as a NumPy `ndarray`
   min_index = np.argmin(result.record.energy)
   best_sample = result.record.sample[min_index]

Dense vs Sparse Mode
--------------------

By default the dense interaction matrix is used.  For sparse graphs, pass
``sparse=True`` to take advantage of the CSR-based shared-memory kernel:

.. code-block:: python

   result = sampler.sample_ising(h, J, num_sweeps=1000, sparse=True)

Sparse mode is limited by GPU shared-memory capacity and is best suited
for problems where most variable pairs have no interaction.

Working with Results
--------------------

Results are returned as :class:`dimod.SampleSet` objects.

.. code-block:: python

   # Iterate over all samples
   for sample, energy in result.data(["sample", "energy"]):
       print(sample, energy)

   # Best sample
   best = result.first
   print(best.sample, best.energy)

   # Convert to a Pandas DataFrame
   df = result.to_pandas_dataframe()

Next Steps
----------

- Browse the :doc:`/reference/api/index` for detailed parameter documentation.
- See ``examples/`` in the source tree for runnable scripts.
- Read the :doc:`/architecture/overview` notes for architecture details.
