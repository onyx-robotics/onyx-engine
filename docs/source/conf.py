# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Engine'
copyright = '2025, Onyx Robotics Inc'
author = 'Onyx'
release = '0.1.0'

import os
import sys
sys.path.insert(0, os.path.abspath('../../onyxengine/'))


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser',  # Markdown support - load first
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Supports Google/NumPy docstring formats - must come after autodoc
    'sphinx.ext.viewcode',   # Adds links to source code
    'sphinx.ext.doctest',
    'sphinx_copybutton',
]

templates_path = ['_templates']
exclude_patterns = []

# Master document (root of documentation)
root_doc = 'index'

# -- Options for MyST Parser -------------------------------------------------
# https://myst-parser.readthedocs.io/en/latest/configuration.html

myst_enable_extensions = [
    "colon_fence",  # Allow code fences using ::: syntax
    "deflist",      # Enable definition lists
    "substitution", # Enable substitutions
    "dollarmath",   # Enable dollar math
]

# Ensure MyST parses all reStructuredText directives properly
myst_heading_anchors = 2

# -- Options for autodoc -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration

# Ensure autodoc processes docstrings correctly
autodoc_typehints = "description"  # Show type hints in description, not signature
autodoc_typehints_format = "short"  # Use short names for types
autodoc_member_order = "bysource"   # Order members by source order
autodoc_preserve_defaults = True     # Preserve default argument values

# -- Options for Napoleon (Google/NumPy docstring parser) ----------------
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html#configuration

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_use_ivar = False
napoleon_use_param = True  # Convert Args: to :param: field lists
napoleon_use_rtype = True  # Convert Returns: to :rtype: field lists
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Ensure Napoleon processes docstrings before autodoc formats them
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_admonition_for_warnings = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    "logo_only": True,
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
}
html_static_path = ['_static']
html_css_files = [
    'css/custom.css',
]
html_logo = '_static/engine_logo.svg'

# Ensure proper HTML rendering
html_show_sourcelink = True
html_show_sphinx = True