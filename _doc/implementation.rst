=========================
Comparing implementations
=========================

The following benchmarks compare different implementations
of the same algorithm.

.. contents::
    :local:

Benchmarks around scikit-learn
==============================

.. toctree::
    :maxdepth: 1

    scikit-learn/gridsearch_cache

Some benchmarks are available on :epkg:`PolynomialFeatures`
at `Benchmark of PolynomialFeatures + partialfit of SGDClassifier
<http://www.xavierdupre.fr/app/pymlbenchmark/helpsphinx/gyexamples/
plot_bench_polynomial_features_partial_fit_standalone.html>`_.

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
  <http://www.xavierdupre.fr/app/ensae_teaching_cs/helpsphinx/notebooks/cffi_linear_regression.html>`_

The next benchmark compares the gain obtained by
playing a criterion for decision tree regressor.

* `Custom Criterion for DecisionTreeRegressor <http://www.xavierdupre.fr/app/mlinsights/helpsphinx/notebooks/piecewise_linear_regression_criterion.html>`_
