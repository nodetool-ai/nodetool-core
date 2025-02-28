#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import nodetool

# -- General configuration ---------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"

# General information about the project
project = "Nodetool Core"
copyright = "2024, Matthias Georgi"
author = "Matthias Georgi"
version = "0.6.0"
release = "0.6.0"

language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = "sphinx"
todo_include_todos = False

# -- Options for HTML output -------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Options for HTMLHelp output ---------------------------------------
htmlhelp_basename = "nodetooldoc"

# -- Options for LaTeX output ------------------------------------------
latex_elements = {}
latex_documents = [
    (
        master_doc,
        "nodetool.tex",
        "Nodetool Core Documentation",
        "Matthias Georgi",
        "manual",
    ),
]

# -- Options for manual page output ------------------------------------
man_pages = [(master_doc, "nodetool", "Nodetool Core Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------
texinfo_documents = [
    (
        master_doc,
        "nodetool",
        "Nodetool Core Documentation",
        author,
        "nodetool",
        "Core library for Nodetool, providing functionality for building and running AI workflows.",
        "Miscellaneous",
    ),
]
