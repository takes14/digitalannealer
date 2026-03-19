Samplers
========

.. module:: digitalannealer.samplers

The ``digitalannealer.samplers`` package provides sampler classes for solving
Ising and QUBO optimization problems.  Both samplers share the
:class:`BaseSampler` interface and return :class:`dimod.SampleSet` objects.

DASampler
---------

.. autoclass:: digitalannealer.samplers.da_sampler.DASampler
   :members:
   :undoc-members:
   :show-inheritance:

PTDASampler
-----------

.. autoclass:: digitalannealer.samplers.ptda_sampler.PTDASampler
   :members:
   :undoc-members:
   :show-inheritance:

BaseSampler
-----------

.. autoclass:: digitalannealer.samplers.base_sampler.BaseSampler
   :members:
   :undoc-members:
   :show-inheritance:
