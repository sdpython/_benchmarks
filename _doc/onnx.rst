===============
ONNX benchmarks
===============

The following benchmarks compare runtime or *backend*
with :epkg:`ONNX`.

.. contents::
    :local:

Benchmark for ONNX
==================

The following benchmarks measure the prediction time between
:epkg:`scikit-learn`, :epkg:`onnxruntime` and :epkg:`mlprodict`
for different models related to *one-off* predictions
and *batch* predictions.

.. toctree::
    :maxdepth: 1

    onnx/onnxruntime_datasets_num
    onnx/onnxruntime_datasets_num_reg
    onnx/onnxruntime_datasets_num_reg_knn
    onnx/onnxruntime_ml_ensemble
    onnx/onnxruntime_lr
    onnx/onnxruntime_dt
    onnx/onnxruntime_dt_reg
    onnx/onnxruntime_gbr_reg
    onnx/onnxruntime_gpr_reg
    onnx/onnxruntime_hgb_reg
    onnx/onnxruntime_knn
    onnx/onnxruntime_rf
    onnx/onnxruntime_mlp

Benchmark specific operators (Add, Scaler, ...)
===============================================

The following benchmarks look into simplified models
to help understand how runtime behave for specific operators.

.. toctree::
    :maxdepth: 1

    onnx/onnxruntime_unittest
    onnx/onnxruntime_casc_add
    onnx/onnxruntime_casc_scaler
    onnx/onnxruntime_casc_mlp

Profiling ONNX runtime
======================

.. toctree::
    :maxdepth: 2

    onnx/onnx_profiling

ONNX versus other implementations
=================================

The following pages helps to understand how the runtime
are implemented.


* `Optimisation de code avec cffi, numba, cython
  <http://www.xavierdupre.fr/app/ensae_teaching_cs/helpsphinx3/
  notebooks/cffi_linear_regression.html>`_
* :ref:`l-example-profile`
* :ref:`l-example-onnx-benchmark`
* :ref:`l-example-parallelism`
* :ref:`topkcpprst`
* :ref:`onnxfloatdoubleskldecisiontreesrst`
