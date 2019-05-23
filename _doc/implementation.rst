=========================
Comparing implementations
=========================

The following benchmarks compare different implementations
of the same algorithm.

.. contents::
    :local:

Benchmarks of pull requests on scikit-learn
===========================================

.. toctree::
    :maxdepth: 1

    scikit-learn/PR13290
    scikit-learn/PR13290_2
    scikit-learn/gridsearch_cache

Benchmarks of toy implementations in C++, Python
================================================

The following benchmarks were implemented in other
repositories. The first one measures differents way to write
the dot product in C++ using a couple of processors
optimization such as branching or
`AVX <https://fr.wikipedia.org/wiki/Advanced_Vector_Extensions>`_
instructions.

* `Measures branching in C++ from python
  <http://www.xavierdupre.fr/app/cpyquickhelper/helpsphinx/notebooks/cbenchmark_branching.html>`_
* `Measures a vector sum with different accumulator type
  <http://www.xavierdupre.fr/app/cpyquickhelper/helpsphinx/notebooks/cbenchmark_sum_type.html>`_

The second one looks into the implementation
of a logistic regression with python, C++ or C++ optimization
provided by other libraries.

* `Optimisation de code avec cffi, numba, cython
  <http://www.xavierdupre.fr/app/ensae_teaching_cs/helpsphinx3/notebooks/cffi_linear_regression.html>`_

The next benchmark compares the gain obtained by
playing a criterion for decision tree regressor.

* `Custom Criterion for DecisionTreeRegressor <http://www.xavierdupre.fr/app/mlinsights/helpsphinx/notebooks/piecewise_linear_regression_criterion.html>`_
