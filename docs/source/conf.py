import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = 'calisim'
copyright = '2024, James Bristow'
author = 'James Bristow'
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinxarg.ext",
]

numpydoc_show_class_members = False

templates_path = ["_templates"]

source_suffix = ".rst"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"

html_static_path = ["_static"]

# The master toctree document.
master_doc = "index"

# Datetime

today_fmt = "%d/%m/%y"
