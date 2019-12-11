
.. _l-bench-plot-onnxruntime-random-forest:

Prediction time scikit-learn / onnxruntime for RandomForestClassifier
=====================================================================

.. contents::
    :local:

.. index:: onnxruntime, RandomForestClassifier

Overview
++++++++

.. plot::

    def label_fct(la):
        la = la.replace("onxpython_compiled", "opy")
        la = la.replace("onxonnxruntime1", "ort")
        la = la.replace("True", "1")
        la = la.replace("False", "0")
        la = la.replace("max_depth", "mxd")
        la = la.replace("method=predict_proba", "prob")
        la = la.replace("method=predict", "cl")
        la = la.replace("n_estimators=", "nt=")
        return la

    import matplotlib.pyplot as plt
    import pandas
    from pymlbenchmark.benchmark.bench_helper import bench_pivot
    from pymlbenchmark.plotting import plot_bench_xtime

    name = "../../onnx/results/bench_plot_onnxruntime_random_forest.perf.csv"
    df = pandas.read_csv(name)

    plot_bench_xtime(df, row_cols='N', col_cols='method',
                     hue_cols=['n_estimators'],
                     cmp_col_values=('lib', 'skl'),
                     x_value='mean', y_value='xtime',
                     parallel=(1., 0.5), title=None,
                     ax=None, box_side=4, label_fct=label_fct)
    plt.suptitle("Acceleration onnxruntime / scikit-learn for RandomForestClassifier")
    plt.show()

.. plot::

    def label_fct(la):
        la = la.replace("onxpython_compiled", "opy")
        la = la.replace("onxonnxruntime1", "ort")
        la = la.replace("True", "1")
        la = la.replace("False", "0")
        la = la.replace("max_depth", "mxd")
        la = la.replace("method=predict_proba", "prob")
        la = la.replace("method=predict", "cl")
        la = la.replace("n_estimators=", "nt=")
        return la

    import matplotlib.pyplot as plt
    import pandas
    from pymlbenchmark.plotting import plot_bench_xtime

    name = "../../onnx/results/bench_plot_onnxruntime_random_forest.perf.csv"
    df = pandas.read_csv(name)

    plot_bench_xtime(df, row_cols='N', col_cols='method',
                     hue_cols=['max_depth'],
                     cmp_col_values=('lib', 'skl'),
                     x_value='mean', y_value='xtime',
                     parallel=(1., 0.5), title=None,
                     ax=None, box_side=4, label_fct=label_fct)
    plt.suptitle("Acceleration onnxruntime / scikit-learn for RandomForestClassifier")
    plt.show()

Detailed graphs
+++++++++++++++

.. plot::

    import matplotlib.pyplot as plt
    import pandas
    from pymlbenchmark.plotting import plot_bench_results

    name = "../../onnx/results/bench_plot_onnxruntime_random_forest.perf.csv"
    df = pandas.read_csv(name)

    def label_fct(la):
        la = la.replace("onxpython_compiled", "opy")
        la = la.replace("onxonnxruntime1", "ort")
        la = la.replace("True", "1")
        la = la.replace("False", "0")
        la = la.replace("max_depth", "mxd")
        la = la.replace("method=predict_proba", "prob")
        la = la.replace("method=predict", "cl")
        la = la.replace("n_estimators=", "nt=")
        return la

    plot_bench_results(df, row_cols=['onnx_options', 'N', 'n_estimators'],
                       col_cols='method',
                       hue_cols='max_depth',
                       cmp_col_values=('lib', 'skl'),
                       x_value='dim', y_value='mean',
                       title=None, label_fct=label_fct,
                       ax=None, box_side=4)
    plt.suptitle("Acceleration onnxruntime / scikit-learn for RandomForestClassifier")
    plt.show()

Configuration
+++++++++++++

.. runpython::
    :rst:
    :warningout: RuntimeWarning
    :showcode:

    from pyquickhelper.pandashelper import df2rst
    import pandas
    name = os.path.join(__WD__, "../../onnx/results/bench_plot_onnxruntime_random_forest.time.csv")
    df = pandas.read_csv(name)
    print(df2rst(df, number_format=4))

Raw results
+++++++++++

:download:`bench_plot_onnxruntime_random_forest.csv <../../onnx/results/bench_plot_onnxruntime_random_forest.perf.csv>`

.. runpython::
    :rst:
    :warningout: RuntimeWarning
    :showcode:
    :toggle: out

    from pyquickhelper.pandashelper import df2rst
    from pymlbenchmark.benchmark.bench_helper import bench_pivot
    import pandas
    name = os.path.join(__WD__, "../../onnx/results/bench_plot_onnxruntime_random_forest.perf.csv")
    df = pandas.read_csv(name)
    piv = bench_pivot(df).reset_index(drop=False)
    piv['speedup_py'] = piv['skl'] / piv['onxpython_compiled']
    piv['speedup_ort'] = piv['skl'] / piv['onxonnxruntime1']
    print(df2rst(piv, number_format=4))

.. runpython::
    :rst:
    :warningout: RuntimeWarning
    :showcode:
    :toggle: out

    from pyquickhelper.pandashelper import df2rst
    import pandas
    name = os.path.join(__WD__, "../../onnx/results/bench_plot_onnxruntime_random_forest.perf.csv")
    df = pandas.read_csv(name)
    df = df[df['lib'] == 'skl']
    print(df2rst(df, number_format=4))

Benchmark code
++++++++++++++

`bench_plot_onnxruntime_random_forest.py <https://github.com/sdpython/_benchmarks/blob/master/onnx/bench_plot_onnxruntime_random_forest.py>`_

.. literalinclude:: ../../onnx/bench_plot_onnxruntime_random_forest.py
    :language: python
