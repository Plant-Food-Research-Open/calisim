import os
import sys

import calisim  # noqa: F401

sys.path.insert(0, os.path.abspath("../.."))

project = "calisim"
copyright = "2024, James Bristow"
author = "James Bristow"
release = "0.1.0"

extensions = [
	"sphinx.ext.napoleon",
	"sphinx.ext.autodoc",
	"sphinx.ext.intersphinx",
	"sphinx.ext.ifconfig",
	"sphinx.ext.viewcode",
	"sphinx.ext.githubpages",
	"sphinxarg.ext",
	"sphinxcontrib.autodoc_pydantic",
	"myst_nb",
]

numpydoc_show_class_members = False

templates_path = ["_templates"]

source_suffix = ".rst"

html_theme = "sphinx_rtd_theme"

html_static_path = ["_static"]

master_doc = "index"

today_fmt = "%d/%m/%y"

smv_branch_whitelist = "main"
smv_remote_whitelist: str | None = None
smv_released_pattern = r"^refs/tags/.*$"

autodoc_pydantic_model_show_json = False

intersphinx_mapping = {
	"sklearn": ("https://scikit-learn.org/stable/", None),
}
