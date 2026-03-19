"""Pytest configuration."""

import importlib.util

import pytest


def cuda_backend_available():
    """Check if CUDA backend is available.

    Returns:
        bool: True if the _core module with CUDA support is available.
    """
    spec = importlib.util.find_spec("digitalannealer._core")
    if spec is None:
        return False
    try:
        import digitalannealer._core  # noqa: F401
    except Exception:
        return False
    return True


# Make available as fixture and module-level flag
CUDA_AVAILABLE = cuda_backend_available()


@pytest.fixture
def requires_cuda():
    """Fixture to skip tests that require CUDA."""
    if not CUDA_AVAILABLE:
        pytest.skip("CUDA backend not available")
