
.. _l-bench-plot-onnxruntime-knn:

Prediction time scikit-learn / onnxruntime for KNeighborsClassifier
===================================================================

.. contents::
    :local:

.. index:: onnxruntime, KNeighborsClassifier

Overview
++++++++

.. plot::

    import matplotlib.pyplot as plt
    import pandas
    from pymlbenchmark.benchmark.bench_helper import bench_pivot
    from pymlbenchmark.plotting import plot_bench_xtime

    name = "../../onnx/results/bench_plot_onnxruntime_knn.perf.csv"
    df = pandas.read_csv(name)

    plot_bench_xtime(df, row_cols=['N', 'leaf_size'], col_cols='method',
                     hue_cols=['metric'],
                     cmp_col_values=('lib', 'skl'),
                     x_value='mean', y_value='xtime',
                     parallel=(1., 0.5), title=None,
                     ax=None, box_side=4)
    plt.suptitle("Acceleration onnxruntime / scikit-learn for KNeighborsClassifier")
    plt.show()

.. plot::

    import matplotlib.pyplot as plt
    import pandas
    from pymlbenchmark.plotting import plot_bench_xtime

    name = "../../onnx/results/bench_plot_onnxruntime_knn.perf.csv"
    df = pandas.read_csv(name)

    plot_bench_xtime(df, row_cols=['N', 'leaf_size'], col_cols='method',
                     hue_cols=['n_neighbors'],
                     cmp_col_values=('lib', 'skl'),
                     x_value='mean', y_value='xtime',
                     parallel=(1., 0.5), title=None,
                     ax=None, box_side=4)
    plt.suptitle("Acceleration onnxruntime / scikit-learn for KNeighborsClassifier")
    plt.show()

Detailed graphs
+++++++++++++++

.. plot::

    import matplotlib.pyplot as plt
    import pandas
    from pymlbenchmark.plotting import plot_bench_results

    name = "../../onnx/results/bench_plot_onnxruntime_knn.perf.csv"
    df = pandas.read_csv(name)

    plot_bench_results(df, row_cols=['N', 'n_neighbors'], col_cols='method',
                       x_value='dim', hue_cols='metric',
                       title=None,
                       ax=None, box_side=4)
    plt.suptitle("Acceleration onnxruntime / scikit-learn for KNeighborsClassifier")
    plt.show()

Configuration
+++++++++++++

.. runpython::
    :rst:
    :warningout: RuntimeWarning
    :showcode:

    from pyquickhelper.pandashelper import df2rst
    import pandas
    name = os.path.join(__WD__, "../../onnx/results/bench_plot_onnxruntime_knn.time.csv")
    df = pandas.read_csv(name)
    print(df2rst(df, number_format=4))

Raw results
+++++++++++

:download:`bench_plot_onnxruntime_knn.csv <../../onnx/results/bench_plot_onnxruntime_knn.perf.csv>`

.. runpython::
    :rst:
    :warningout: RuntimeWarning
    :showcode:
    :toggle: out

    from pyquickhelper.pandashelper import df2rst
    from pymlbenchmark.benchmark.bench_helper import bench_pivot
    import pandas
    name = os.path.join(__WD__, "../../onnx/results/bench_plot_onnxruntime_knn.perf.csv")
    df = pandas.read_csv(name)
    piv = bench_pivot(df).reset_index(drop=False)
    piv['speedup_py'] = piv['skl'] / piv['onxpython']
    piv['speedup_ort'] = piv['skl'] / piv['onxonnxruntime1']
    print(df2rst(piv, number_format=4))

Benchmark code
++++++++++++++

`bench_plot_onnxruntime_knn.py <https://github.com/sdpython/_benchmarks/blob/master/onnx/bench_plot_onnxruntime_knn.py>`_

.. literalinclude:: ../../onnx/bench_plot_onnxruntime_knn.py
    :language: python
