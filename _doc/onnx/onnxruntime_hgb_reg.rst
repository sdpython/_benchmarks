
.. _l-bench-plot-onnxruntime-hgb-reg:

Benchmark (ONNX) for HistBoostingGradientRegressor
==================================================

.. contents::
    :local:

.. index:: onnxruntime, HistBoostingGradientRegressor

Overview
++++++++

.. plot::

    def label_fct(la):
        la = la.replace("onxpython_compiled", "opy")
        la = la.replace("onxpython", "opy")
        la = la.replace("onxonnxruntime1", "ort")
        la = la.replace("True", "1")
        la = la.replace("False", "0")
        la = la.replace("max_depth", "mxd")
        la = la.replace("method=predict", "cl")
        la = la.replace("method=proba", "prob")
        return la

    import matplotlib.pyplot as plt
    import pandas
    from pymlbenchmark.plotting import plot_bench_xtime

    name = "../../onnx/results/bench_plot_onnxruntime_hgb.perf.csv"
    df = pandas.read_csv(name)

    plot_bench_xtime(df, row_cols='N', col_cols='max_depth',
                     hue_cols='method',
                     cmp_col_values=('lib', 'skl'),
                     x_value='mean', y_value='xtime',
                     parallel=(1., 0.5), title=None, fontsize=24,
                     ax=None, box_side=4, label_fct=label_fct)
    plt.suptitle("Acceleration onnxruntime / scikit-learn for DecisionTreeRegressor")
    plt.show()

Detailed graphs
+++++++++++++++

.. plot::

    def label_fct(la):
        la = la.replace("onxpython_compiled", "opy")
        la = la.replace("onxpython", "opy")
        la = la.replace("onxonnxruntime1", "ort")
        la = la.replace("True", "1")
        la = la.replace("False", "0")
        la = la.replace("max_depth", "mxd")
        la = la.replace("method=predict", "cl")
        la = la.replace("method=proba", "prob")
        return la

    import matplotlib.pyplot as plt
    import pandas
    from pymlbenchmark.plotting import plot_bench_results

    name = "../../onnx/results/bench_plot_onnxruntime_hgb.perf.csv"
    df = pandas.read_csv(name)

    plot_bench_results(df, row_cols=('N', 'n_estimators'), col_cols='max_depth',
                       hue_cols='method',
                       cmp_col_values=('lib', 'skl'),
                       x_value='dim', y_value='mean',
                       title=None, label_fct=label_fct, fontsize=12,
                       ax=None, box_side=4)
    plt.suptitle("Acceleration onnxruntime / scikit-learn for HistBoostingGradientRegressor")
    plt.show()

Configuration
+++++++++++++

.. runpython::
    :rst:
    :warningout: RuntimeWarning
    :showcode:

    from pyquickhelper.pandashelper import df2rst
    import pandas
    name = os.path.join(__WD__, "../../onnx/results/bench_plot_onnxruntime_hgb.time.csv")
    df = pandas.read_csv(name)
    print(df2rst(df, number_format=4))

Raw results
+++++++++++

:download:`bench_plot_onnxruntime_hgb.csv <../../onnx/results/bench_plot_onnxruntime_hgb.perf.csv>`

.. runpython::
    :rst:
    :warningout: RuntimeWarning
    :showcode:
    :toggle: out

    from pyquickhelper.pandashelper import df2rst
    from pymlbenchmark.benchmark.bench_helper import bench_pivot
    import pandas
    name = os.path.join(__WD__, "../../onnx/results/bench_plot_onnxruntime_hgb.perf.csv")
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
    name = os.path.join(__WD__, "../../onnx/results/bench_plot_onnxruntime_hgb.perf.csv")
    df = pandas.read_csv(name)
    df = df[df['lib'] == 'skl']
    print(df2rst(df, number_format=4))

Benchmark code
++++++++++++++

`bench_plot_onnxruntime_hgb.py
<https://github.com/sdpython/_benchmarks/blob/master/onnx/bench_plot_onnxruntime_hgb.py>`_

.. literalinclude:: ../../onnx/bench_plot_onnxruntime_hgb.py
    :language: python
