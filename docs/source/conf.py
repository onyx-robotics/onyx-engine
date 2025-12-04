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
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Supports Google/NumPy docstring formats
    'sphinx.ext.viewcode',   # Adds links to source code
    'sphinx.ext.doctest',
    'sphinx_copybutton',
    'myst_parser',  # Markdown support
]

templates_path = ['_templates']
exclude_patterns = []

# Master document (root of documentation)
root_doc = 'index'

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