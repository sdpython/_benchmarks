# -*- coding: utf-8 -*-
# Licensed under the MIT License.

import os
import sys
import shutil
import sphinx_gallery.gen_gallery
import sphinx_modern_theme_modified
this = os.path.abspath(os.path.dirname(__file__))
new_paths = [os.path.join(this, '..', 'scikit-learn', 'results')]
for np in new_paths:
    assert os.path.exists(np)
    sys.path.append(np)


# -- Project information -----------------------------------------------------

project = 'Benchmarks about Machine Learning'
copyright = '2019, Xavier Dupré'
author = 'Xavier Dupré'
version = '0.1'
release = version

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.imgmath',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    "sphinx.ext.autodoc",
    'sphinx.ext.githubpages',
    # "sphinx_gallery.gen_gallery",
    'sphinx.ext.autodoc',
    "sphinxcontrib.blockdiag",
    "pyquickhelper.sphinxext.sphinx_runpython_extension",
    "pyquickhelper.sphinxext.sphinx_epkg_extension",
    "pyquickhelper.sphinxext.sphinx_collapse_extension",
]

try:
    import matplotlib.sphinxext
    assert matplotlib.sphinxext is not None
    extensions.append('matplotlib.sphinxext.plot_directive')
except ImportError:
    # matplotlib is not installed.
    pass

templates_path = ['_templates']
source_suffix = ['.rst']

master_doc = 'index'
language = "en"
exclude_patterns = []
pygments_style = 'default'

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_mo"
html_static_path = ['_static']
html_theme = "sphinx_modern_theme_modified"
html_theme_path = [sphinx_modern_theme_modified.get_html_theme_path()]
html_logo = "logo_main.png"

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {'https://docs.python.org/': None}

# -- Options for Sphinx Gallery ----------------------------------------------

sphinx_gallery_conf = {
    'examples_dirs': 'examples',
    'gallery_dirs': 'auto_examples',
}

# -- shortcuts ---------------------------------------------------------------

epkg_dictionary = {
    'onnx': 'https://github.com/onnx/onnx',
    'ONNX': 'https://onnx.ai/',
    'onnxruntime': 'https://github.com/Microsoft/onnxruntime',
    'PolynomialFeatures': 'https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html',
    'Python': 'https://www.python.org/',
    'scikit-learn': 'https://scikit-learn.org/stable/',
}

# -- Setup actions -----------------------------------------------------------


def setup(app):
    # Placeholder to initialize the folder before
    # generating the documentation.
    return app
