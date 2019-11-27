===============
ONNX benchmarks
===============

The following benchmarks compare runtime or *backend*
with :epkg:`ONNX`.

.. contents::
    :local:

onxruntime VS scikit-learn
==========================

The following benchmark measures the prediction time between
:epkg:`scikit-learn` and :epkg:`onnxruntime` for different models
related to *one-off* predictions and *batch* predictions.
:epkg:`onnxruntime` allows for some models to run batch
predictions by using broadcasting. It is not available
for all models.

.. toctree::
    :maxdepth: 1

    onnx/onnxruntime_datasets_num
    onnx/onnxruntime_lr
    onnx/onnxruntime_dt
    onnx/onnxruntime_dt_reg
    onnx/onnxruntime_knn
    onnx/onnxruntime_rf
    onnx/onnxruntime_mlp

onnxruntime specific configurations
===================================

The following benchmark look into simplified models
to help understand how :epkg:`onnxruntime` works.

.. toctree::
    :maxdepth: 1

    onnx/onnxruntime_unittest
    onnx/onnxruntime_casc_add
    onnx/onnxruntime_casc_scaler
    onnx/onnxruntime_casc_mlp

ONNX versus other implementations
=================================

The following benchmarks were implemented in other
repositories. The first one looks into the implementation
of a logistic regression with python, C++ or C++ optimization
provided by other libraries.

* `Optimisation de code avec cffi, numba, cython
  <http://www.xavierdupre.fr/app/ensae_teaching_cs/helpsphinx3/notebooks/cffi_linear_regression.html>`_
