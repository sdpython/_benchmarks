
.. _l-bench-plot-onnxruntime-cascade-mlp:

Prediction time for intermediate outputs
========================================

.. contents::
    :local:

.. index:: onnxruntime, MLPClassifier

The experiment compares the execution time between
for all intermediate nodes. The original graph
is truncated from the input node to every intermediate node.
The network was trained on a binary classification,
it has 10 features, two layers with 10 networks each.

.. toctree::

    onnxruntime_casc_mlp_img

Overview
++++++++

.. plot::

    import matplotlib.pyplot as plt
    import pandas

    name = "../../onnx/results/bench_plot_onnxruntime_casc_mlp.perf.csv"
    df = pandas.read_csv(name)
    data = df[["N", "mean", "method"]].pivot('N', 'method', 'mean')
    for c in data.columns:
        data[c] /= data['skl_proba']

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    data[['skl_proba', 'onnx_proba']].plot(ax=ax[0]);
    data.plot(ax=ax[1]);

    plt.suptitle("Speed against scikit-learn, lower is faster.")
    plt.show()

Configuration
+++++++++++++

.. runpython::
    :rst:
    :warningout: RuntimeWarning
    :showcode:

    from pyquickhelper.pandashelper import df2rst
    import pandas
    name = os.path.join(__WD__, "../../onnx/results/bench_plot_onnxruntime_casc_mlp.time.csv")
    df = pandas.read_csv(name)
    print(df2rst(df, number_format=4))

Raw results
+++++++++++

:download:`bench_plot_onnxruntime_casc_mlp.csv <../../onnx/results/bench_plot_onnxruntime_casc_mlp.perf.csv>`

.. runpython::
    :rst:
    :warningout: RuntimeWarning
    :showcode:
    :toggle: out

    from pyquickhelper.pandashelper import df2rst
    from pymlbenchmark.benchmark.bench_helper import bench_pivot
    import pandas
    name = os.path.join(__WD__, "../../onnx/results/bench_plot_onnxruntime_casc_mlp.perf.csv")
    df = pandas.read_csv(name)
    piv = bench_pivot(df).reset_index(drop=False)
    piv['speedup'] = piv['skl'] / piv['ort']
    print(df2rst(piv, number_format=4))

Benchmark code
++++++++++++++

`bench_plot_onnxruntime_casc_mlp.py <https://github.com/sdpython/_benchmarks/blob/master/onnx/bench_plot_onnxruntime_casc_mlp.py>`_

.. literalinclude:: ../../onnx/bench_plot_onnxruntime_casc_mlp.py
    :language: python
