
.. _l-bench-plot-onnxruntime-logreg:

Prediction time scikit-learn / onnxruntime for LogisticRegression
=================================================================

.. contents::
    :local:

.. index:: onnxruntime, LogisticRegression

Overview
++++++++

.. plot::

    import matplotlib.pyplot as plt
    import pandas
    from pymlbenchmark.plotting import plot_bench_xtime

    name = "../../onnx/results/bench_plot_onnxruntime_logreg.perf.csv"
    df = pandas.read_csv(name)

    plot_bench_xtime(df, row_cols='N', col_cols='method',
                     hue_cols='fit_intercept',
                     cmp_col_values=('lib', 'skl'),
                     x_value='mean', y_value='xtime',
                     parallel=(1., 0.5), title=None,
                     ax=None, box_side=4)
    plt.suptitle("Acceleration onnxruntime / scikit-learn for LogisticRegression")
    plt.show()

Detailed graphs
+++++++++++++++

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
        la = la.replace("fit_intercept=1", "+biais")
        la = la.replace("fit_intercept=True", "+biais")
        return la

    import matplotlib.pyplot as plt
    import pandas
    from pymlbenchmark.plotting import plot_bench_results

    name = "../../onnx/results/bench_plot_onnxruntime_logreg.perf.csv"
    df = pandas.read_csv(name)

    plot_bench_results(df, row_cols=['N', 'onnx_options'], col_cols='method',
                              hue_cols='fit_intercept',
                     cmp_col_values=('lib', 'skl'),
                     x_value='dim', y_value='mean',
                     title=None, label_fct=label_fct,
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
    from pymlbenchmark.benchmark.bench_helper import bench_pivot
    import pandas
    name = os.path.join(__WD__, "../../onnx/results/bench_plot_onnxruntime_logreg.perf.csv")
    df = pandas.read_csv(name)
    piv = bench_pivot(df).reset_index(drop=False)
    piv['speedup_py'] = piv['skl'] / piv['onxpython_compiled']
    piv['speedup_ort'] = piv['skl'] / piv['onxonnxruntime1']
    print(df2rst(piv, number_format=4))

Benchmark code
++++++++++++++

`bench_plot_onnxruntime_logreg.py <https://github.com/sdpython/_benchmarks/blob/master/onnx/bench_plot_onnxruntime_logreg.py>`_

.. literalinclude:: ../../onnx/bench_plot_onnxruntime_logreg.py
    :language: python
