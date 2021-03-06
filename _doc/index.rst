
=================================
Benchmarks about Machine Learning
=================================

This project started with my first attempt
to bring a modification to :epkg:`scikit-learn`.
My first `pull request <https://github.com/scikit-learn/scikit-learn/pull/13290>`_
was about optimizing the computation of polynomial features.
I reused the template to measure various implementations
or models.

.. toctree::
    :maxdepth: 1

    implementation
    modules
    onnx
    technical
    sklbench
    glossary

Others benchmarks with :epkg:`asv` and :epkg:`onnx`:

* `Scikit-Learn/ONNX benchmark with AirSpeedVelocity (official)
  <../../benches/scikit-learn_benchmarks/index.html>`_
* `Prediction with scikit-learn and ONNX benchmark
  <../../benches/asv-skl2onnx/index.html>`_
  (or subset `Prediction with scikit-learn and ONNX benchmark (SVM + Trees)
  <../../benches/asv-skl2onnx-cpp/index.html>`_)

Smaller benchmarks:

* `mlprodict model of benchmark
  <../../mlprodict_bench/helpsphinx//index.html>`_
* `mlprodict model applied to linear models
  <../../mlprodict_bench2/helpsphinx/index.html>`_

*Links:*

* `github <https://github.com/sdpython/_benchmarks/>`_,
* :ref:`genindex`
