Architecture Overview
=====================

This section summarises the high-level architecture of the Digital Annealer
library.  For the full design documents see the source-tree files listed below.
The research background papers used for this implementation are listed in
:doc:`/project/references`.

System Structure
----------------

The system follows a **Core Engine + Python Binding** split:

1. **Python Frontend** — validates and normalises user input
   (``digitalannealer.utils``), builds the annealing schedule, and returns
   ``dimod.SampleSet`` results.
2. **PyBind11 Bindings** — convert Python/NumPy objects to C++ types and
   release the GIL so the GPU work can proceed without blocking the
   interpreter.
3. **C++ Engine** — orchestrates the annealing schedule, manages device
   memory, and launches CUDA kernels.
4. **CUDA Kernels** — execute spin updates, parallel trial selection,
   dynamic offset adjustment, and replica exchange entirely on the GPU.

Execution Model
---------------

The host builds a ``(beta, duration)`` schedule once per ``sample_ising``
call, uploads it to VRAM, and launches a **single monolithic kernel** for
the full ``num_sweeps`` loop.  There are no per-sweep kernel launches from
Python.

Parallel Tempering
------------------

Parallel Tempering runs multiple replicas at different inverse temperatures.
Replica exchange uses **zero-copy swaps** via an index map — spin buffers
are never moved.  Exchange logic stays entirely inside the GPU kernel.

Dense vs Sparse Kernels
-----------------------

- **Dense** — full :math:`N \times N` interaction matrix; best for
  fully-connected or near-fully-connected graphs.
- **Sparse** — CSR format with a shared-memory resident path; efficient
  when the number of interactions is much smaller than :math:`N^2`.
  The runtime may override a sparse request when the problem exceeds
  shared-memory limits.

Reference Documents
-------------------

The following design documents are maintained alongside this documentation:

- :doc:`basic_design_document` — high-level architecture
- :doc:`detailed_design_document` — component-level design
- :doc:`requirements_definition_document` — requirements and scope
