Utilities
=========

.. module:: digitalannealer.utils

Validation, normalization and energy computation helpers used internally
by the samplers.  These are re-exported from ``digitalannealer.utils`` for
convenience.

Validation
----------

.. autofunction:: digitalannealer.utils.validation_input.validate_ising_input

.. autofunction:: digitalannealer.utils.validation_input.validate_qubo_input

.. autofunction:: digitalannealer.utils.validation_input.validate_dtype

Normalization
-------------

.. autofunction:: digitalannealer.utils.normalization.normalize_ising_to_dense

.. autofunction:: digitalannealer.utils.normalization.normalize_ising_to_sparse

CSR Validation
--------------

.. autofunction:: digitalannealer.utils.csr_validation.validate_csr_structure

Energy Computation
------------------

.. autofunction:: digitalannealer.utils.energy.compute_all_delta_e

.. autofunction:: digitalannealer.utils.energy.update_delta_e

.. autofunction:: digitalannealer.utils.energy.compute_energy
