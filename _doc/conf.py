# -*- coding: utf-8 -*-
# Licensed under the MIT License.

import os
import sys
import shutil
import sphinx_gallery.gen_gallery
import alabaster

this = os.path.abspath(os.path.dirname(__file__))
new_paths = [os.path.join(this, '..', 'scikit-learn', 'results')]
for np in new_paths:
    assert os.path.exists(np)
    sys.path.append(np)


# -- Project information -----------------------------------------------------

project = 'Benchmarks about Machine Learning'
copyright = '2020, Xavier Dupré'
author = 'Xavier Dupré'
version = '0.2'
release = version

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.imgmath',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    "sphinx.ext.autodoc",
    'sphinx.ext.githubpages',
    "sphinx_gallery.gen_gallery",
    'sphinx.ext.autodoc',
    "sphinxcontrib.blockdiag",
    "pyquickhelper.sphinxext.sphinx_runpython_extension",
    "pyquickhelper.sphinxext.sphinx_epkg_extension",
    "pyquickhelper.sphinxext.sphinx_collapse_extension",
    "pyquickhelper.sphinxext.sphinx_postcontents_extension",
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

html_static_path = ['_static']
html_theme = "alabaster"
html_theme_path = [alabaster.get_path()]
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
    'asv': 'https://github.com/airspeed-velocity/asv',
    'BLAS': 'https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms',
    'build_custom_scenarios': 'http://www.xavierdupre.fr/app/mlprodict/helpsphinx/mlprodict/onnxrt/validate/validate_scenarios.html?highlight=build_custom_scenarios#mlprodict.onnxrt.validate.validate_scenarios.build_custom_scenarios',
    'find_suitable_problem': 'http://www.xavierdupre.fr/app/mlprodict/helpsphinx/mlprodict/onnxrt/validate/validate_problems.html#mlprodict.onnxrt.validate.validate_problems.find_suitable_problem',
    'joblib': 'https://joblib.readthedocs.io/en/latest/',
    'mlprodict': 'http://www.xavierdupre.fr/app/mlprodict/helpsphinx/index.html',
    'numpy': 'https://www.numpy.org/',
    'onnx': 'https://github.com/onnx/onnx',
    'ONNX': 'https://onnx.ai/',
    'onnxruntime': 'https://github.com/Microsoft/onnxruntime',
    'PolynomialFeatures': 'https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html',
    'py-spy': 'https://github.com/benfred/py-spy',
    'pyrtc': 'http://www.xavierdupre.fr/app/mlprodict/helpsphinx/onnx_runtime.html#python-compiled',
    'Python': 'https://www.python.org/',
    'scikit-learn': 'https://scikit-learn.org/stable/',
    'scikit-learn_benchmarks': 'https://github.com/jeremiedbb/scikit-learn_benchmarks',
    'sklearn-onnx': 'https://github.com/onnx/sklearn-onnx',
}

# -- Setup actions -----------------------------------------------------------


def setup(app):
    # Placeholder to initialize the folder before
    # generating the documentation.
    return app
