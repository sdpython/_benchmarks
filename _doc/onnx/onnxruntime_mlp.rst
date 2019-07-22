
.. _l-bench-plot-onnxruntime-mlpclassifier:

Prediction time scikit-learn / onnxruntime for MLPClassifier
============================================================

.. contents::
    :local:

.. index:: onnxruntime, MLPClassifier

Overview
++++++++

.. plot::

    import matplotlib.pyplot as plt
    import pandas
    from pymlbenchmark.benchmark.bench_helper import bench_pivot
    from pymlbenchmark.plotting import plot_bench_xtime

    name = "../../onnx/results/bench_plot_onnxruntime_mlp.perf.csv"
    df = pandas.read_csv(name)

    def label_fct(la):
        la = la.replace("onxpython", "opy")
        la = la.replace("onxonnxruntime1", "ort")
        la = la.replace("fit_intercept", "fi")
        la = la.replace("True", "1")
        la = la.replace("False", "0")
        la = la.replace("activation=", "")
        return la

    plot_bench_xtime(df, row_cols=['N', 'hidden_layer_sizes'],
                     col_cols='method',
                     hue_cols=['activation'],
                     cmp_col_values=('lib', 'skl'),
                     x_value='mean', y_value='xtime',
                     label_fct=label_fct,
                     parallel=(1., 2.))

    plt.suptitle("Acceleration onnxruntime / scikit-learn for MLPClassifier")
    plt.show()

Detailed graphs
+++++++++++++++

.. plot::

    def label_fct(la):
        la = la.replace("onxpython", "opy")
        la = la.replace("onxonnxruntime1", "ort")
        la = la.replace("fit_intercept", "fi")
        la = la.replace("True", "1")
        la = la.replace("False", "0")
        la = la.replace("activation=", "")
        la = la.replace("logistic", "logc")
        return la

    import matplotlib.pyplot as plt
    import pandas
    from pymlbenchmark.plotting import plot_bench_results

    name = "../../onnx/results/bench_plot_onnxruntime_mlp.perf.csv"
    df = pandas.read_csv(name)

    plot_bench_results(df, row_cols=['N', 'hidden_layer_sizes'],
                       col_cols='method',
                       hue_cols='activation',
                       cmp_col_values=('lib', 'skl'),
                       x_value='dim', y_value='mean',
                       title=None, label_fct=label_fct,
                       ax=None, box_side=4)
    plt.suptitle("Acceleration onnxruntime / scikit-learn for MLPClassifier")
    plt.show()

Configuration
+++++++++++++

.. runpython::
    :rst:
    :warningout: RuntimeWarning
    :showcode:

    from pyquickhelper.pandashelper import df2rst
    import pandas
    name = os.path.join(__WD__, "../../onnx/results/bench_plot_onnxruntime_mlp.time.csv")
    df = pandas.read_csv(name)
    print(df2rst(df, number_format=4))

Raw results
+++++++++++

:download:`bench_plot_onnxruntime_mlp.csv <../../onnx/results/bench_plot_onnxruntime_mlp.perf.csv>`

.. runpython::
    :rst:
    :warningout: RuntimeWarning
    :showcode:
    :toggle: out

    from pyquickhelper.pandashelper import df2rst
    from pymlbenchmark.benchmark.bench_helper import bench_pivot
    import pandas
    name = os.path.join(__WD__, "../../onnx/results/bench_plot_onnxruntime_mlp.perf.csv")
    df = pandas.read_csv(name)
    piv = bench_pivot(df).reset_index(drop=False)
    piv['speedup'] = piv['skl'] / piv['ort']
    print(df2rst(piv, number_format=4))

Benchmark code
++++++++++++++

`bench_plot_onnxruntime_mlp.py <https://github.com/sdpython/_benchmarks/blob/master/onnx/bench_plot_onnxruntime_mlp.py>`_

.. literalinclude:: ../../onnx/bench_plot_onnxruntime_mlp.py
    :language: python
