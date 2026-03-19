# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------
# Add the project root so autodoc can find the digitalannealer package.
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
project = "Digital Annealer"
copyright = "2026, Takenori Sumi"
author = "Takenori Sumi"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for autodoc -----------------------------------------------------
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Mock the compiled C++/CUDA extension so that Sphinx can build the docs
# without a working CUDA toolkit (e.g. on ReadTheDocs).
autodoc_mock_imports = ["digitalannealer._core"]

# -- Options for Napoleon (Google/NumPy style docstrings) --------------------
napoleon_google_docstrings = True
napoleon_numpy_docstrings = True
napoleon_include_init_with_doc = True

# -- Options for intersphinx -------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "dimod": ("https://docs.dwavequantum.com/en/latest/", None),
}

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- GitHub integration ------------------------------------------------------
html_context = {
    "display_github": True,
    "github_user": "takes14",
    "github_repo": "digitalannealer",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
}
