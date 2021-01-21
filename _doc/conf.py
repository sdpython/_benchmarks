# -*- coding: utf-8 -*-
# Licensed under the MIT License.

import os
import sys
import shutil
import sphinx_gallery.gen_gallery
import pydata_sphinx_theme

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
html_theme = "pydata_sphinx_theme"
html_theme_path = pydata_sphinx_theme.get_html_theme_path()
html_logo = "logo_main.png"

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'joblib': ('https://joblib.readthedocs.io/en/latest/', None),
    'python': ('https://docs.python.org/{.major}'.format(
        sys.version_info), None),
    'matplotlib': ('https://matplotlib.org/', None),
    'mlinsights': (
        'http://www.xavierdupre.fr/app/mlinsights/helpsphinx/', None),
    'mlprodict': (
        'http://www.xavierdupre.fr/app/mlprodict/helpsphinx/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'pyquickhelper': (
        'http://www.xavierdupre.fr/app/pyquickhelper/helpsphinx/', None),
    'onnxmltools': (
        'http://www.xavierdupre.fr/app/onnxmltools/helpsphinx/index.html',
        None),
    'onnxruntime': (
        'http://www.xavierdupre.fr/app/onnxruntime/helpsphinx/index.html',
        None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'seaborn': ('https://seaborn.pydata.org/', None),
    'scikit-learn': (
        'https://scikit-learn.org/stable/',
        None),
    'sklearn': (
        'https://scikit-learn.org/stable/',
        None),
    'skl2onnx': (
        'http://www.xavierdupre.fr/app/sklearn-onnx/helpsphinx/index.html',
        None),
    'sklearn-onnx': (
        'http://www.xavierdupre.fr/app/sklearn-onnx/helpsphinx/index.html',
        None),
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
