===============
ONNX benchmarks
===============

The following benchmarks compare runtime or *backeend*
with :epkg:`ONNX`.

One-Off predictions
===================

The following benchmark measures the prediction time between
:epkg:`scikit-learn` and :epkg:`onnxruntime` for different configurations
related to *one-off* predictions: predictions are computed
for one observation at a time which is the standard
scenario in a webservice.
:epkg:`onnxruntime` allows for some models to run batch
predictions. If this functionality is available, it is
usually tested for small batches (like 10 observations).

.. toctree::
    :maxdepth: 1

    onnx/summary
    onnx/onnxruntime_lr
    onnx/onnxruntime_dt
    onnx/onnxruntime_knn
    onnx/onnxruntime_rf

ONNX versus other implementations
=================================

The following benchmarks were implemented in other
repositories. The first one looks into the implementation
of a logistic regression with python, C++ or C++ optimization
provided by other libraries.

* `Optimisation de code avec cffi, numba, cython
  <http://www.xavierdupre.fr/app/ensae_teaching_cs/helpsphinx3/notebooks/cffi_linear_regression.html>`_
  