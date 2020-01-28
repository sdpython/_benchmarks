
.. _l-bench-plot-onnxruntime-ml-ensemble:

Benchmark (ONNX) for ensemble models
====================================

.. contents::
    :local:

.. index:: onnxruntime, datasets, ensemble

Overview
++++++++

.. plot::

    import matplotlib.pyplot as plt
    import pandas
    from pymlbenchmark.plotting import plot_bench_results

    def label_fct(la):
        la = la.replace("-lib=", "")
        la = la.replace("rt=", "-")
        return la

    name = "../../onnx/results/bench_plot_ml_ensemble.perf.csv"
    df = pandas.read_csv(name)
    fig, ax = plt.subplots(3, 2, figsize=(12, 5))

    plot_bench_results(df, row_cols=('rt',), col_cols=('dataset', ), label_fct=label_fct,
                       x_value='N', hue_cols=('lib',), cmp_col_values='lib',
                       title="Numerical datasets\nBenchmark scikit-learn, xgboost, lightgbm",
                       ax=ax, fontsize=12)
    fig.show()

Configuration
+++++++++++++

.. runpython::
    :rst:
    :warningout: RuntimeWarning
    :showcode:

    from pyquickhelper.pandashelper import df2rst
    import pandas
    name = os.path.join(__WD__, "../../onnx/results/bench_plot_ml_ensemble.time.csv")
    df = pandas.read_csv(name)
    print(df2rst(df, number_format=4))

Raw results
+++++++++++

:download:`bench_plot_datasets_num.csv <../../onnx/results/bench_plot_ml_ensemble.perf.csv>`

.. runpython::
    :rst:
    :warningout: RuntimeWarning
    :showcode:
    :toggle: out

    from pyquickhelper.pandashelper import df2rst
    from pymlbenchmark.benchmark.bench_helper import bench_pivot
    import pandas
    name = os.path.join(__WD__, "../../onnx/results/bench_plot_ml_ensemble.perf.csv")
    df = pandas.read_csv(name)
    print(df2rst(df, number_format=4))

Benchmark code
++++++++++++++

`bench_plot_datasets_num.py <https://github.com/sdpython/_benchmarks/blob/master/onnx/bench_plot_ml_ensemble.py>`_

.. literalinclude:: ../../onnx/bench_plot_ml_ensemble.py
    :language: python
