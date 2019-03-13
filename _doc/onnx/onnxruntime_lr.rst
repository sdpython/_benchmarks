
.. _l-bench-plot-onnxruntime-logreg:

Prediction time scikit-learn / onnxruntime: LogisticRegression
==============================================================

.. index:: onnxruntime, LogisticRegression

.. contents::
    :local:

Code
++++

`bench_plot_onnxruntime_logreg.py <https://github.com/sdpython/_benchmarks/blob/master/onnx/bench_plot_onnxruntime_logreg.py>`_

Overview
++++++++

.. plot::

    import matplotlib.pyplot as plt
    import pandas
    from pymlbenchmark.benchmark.bench_helper import bench_pivot
    from pymlbenchmark.plotting import plot_bench_xtime

    name = "../../onnx/results/bench_plot_onnxruntime_logreg.perf.csv"
    df = pandas.read_csv(name)
    piv = bench_pivot(df).reset_index(drop=False)

    plot_bench_xtime(df, row_cols='N', col_cols='method',
                     hue_cols='fit_intercept',
                     cmp_col_values=('lib', 'skl'),
                     x_value='mean', y_value='xtime',
                     parallel=(1., 0.5), title=None,
                     ax=None, box_side=4)
    plt.suptitle("Acceleration onnxruntime / scikit-learn for LogisticRegression")
    plt.show()

:epkg:`onnxruntime` is always faster in that particular scenario.

Detailed graphs
+++++++++++++++

.. plot::

    import matplotlib.pyplot as plt
    import pandas
    from pymlbenchmark.benchmark.bench_helper import bench_pivot
    from pymlbenchmark.plotting import plot_bench_results

    name = "../../onnx/results/bench_plot_onnxruntime_logreg.perf.csv"
    df = pandas.read_csv(name)
    piv = bench_pivot(df).reset_index(drop=False)

    plot_bench_results(df, row_cols='N', col_cols='method',
                              hue_cols='fit_intercept',
                     cmp_col_values=('lib', 'skl'),
                     x_value='dim', y_value='mean',
                     title=None,
                     ax=None, box_side=4)
    plt.suptitle("Acceleration onnxruntime / scikit-learn for LogisticRegression")
    plt.show()

Configuration
+++++++++++++

.. runpython::
    :rst:
    :warningout: RuntimeWarning
    :showcode:

    from pyquickhelper.pandashelper import df2rst
    import pandas
    name = os.path.join(__WD__, "../../onnx/results/bench_plot_onnxruntime_logreg.time.csv")
    df = pandas.read_csv(name)
    print(df2rst(df, number_format=4))

Raw results
+++++++++++

:download:`bench_plot_onnxruntime_logreg.csv <../../onnx/results/bench_plot_onnxruntime_logreg.perf.csv>`

.. runpython::
    :rst:
    :warningout: RuntimeWarning
    :showcode:
    :toggle: out

    from pyquickhelper.pandashelper import df2rst
    import pandas
    name = os.path.join(__WD__, "../../onnx/results/bench_plot_onnxruntime_logreg.csv")
    df = pandas.read_csv(name)
    df['speedup'] = df['time_skl'] / df['time_ort']
    print(df2rst(df, number_format=4))

Benchmark code
++++++++++++++

.. literalinclude:: ../../onnx/bench_plot_onnxruntime_logreg.py
    :language: python
