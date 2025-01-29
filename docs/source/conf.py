# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MetaboZen'
copyright = '2025, Gabriel Reder'
author = 'Gabriel Reder'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys
from unittest.mock import MagicMock

class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

# Mock modules
MOCK_MODULES = [
    'numpy',
    'pandas',
    'scipy',
    'sklearn',
    'sklearn.metrics',
    'matplotlib',
    'matplotlib.pyplot',
    'rpy2',
    'pyteomics',
    'tqdm',
    'yaml',
    'xlrd',
    'lxml'
]

sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

sys.path.insert(0, os.path.abspath('../../src'))

autodoc_mock_imports = [
    'numpy',
    'pandas',
    'scipy',
    'scikit-learn',
    'matplotlib',
    'rpy2',
    'pyteomics',
    'tqdm',
    'yaml',
    'xlrd',
    'lxml',
    'ipdb'
]

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'myst_parser',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton'
]


templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
