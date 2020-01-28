
.. _l-bench-plot-onnxruntime-decision-tree:

Benchmark (ONNX) for DecisionTreeClassifier
===========================================

.. contents::
    :local:

.. index:: onnxruntime, DecisionTreeClassifier

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
        la = la.replace("method=predict_proba", "prob")
        la = la.replace("method=predict", "cl")
        return la

    import matplotlib.pyplot as plt
    import pandas
    from pymlbenchmark.plotting import plot_bench_xtime

    name = "../../onnx/results/bench_plot_onnxruntime_decision_tree.perf.csv"
    df = pandas.read_csv(name)

    plot_bench_xtime(df, row_cols='N', col_cols='max_depth',
                     hue_cols='method',
                     cmp_col_values=('lib', 'skl'),
                     x_value='mean', y_value='xtime',
                     parallel=(1., 0.5), title=None, fontsize='large',
                     ax=None, box_side=4, label_fct=label_fct)
    plt.suptitle("Acceleration onnxruntime / scikit-learn for DecisionTreeClassifier")
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
        return la

    import matplotlib.pyplot as plt
    import pandas
    from pymlbenchmark.plotting import plot_bench_results

    name = "../../onnx/results/bench_plot_onnxruntime_decision_tree.perf.csv"
    df = pandas.read_csv(name)

    plot_bench_results(df, row_cols='N', col_cols='max_depth',
                       hue_cols='method',
                       cmp_col_values=('lib', 'skl'),
                       x_value='dim', y_value='mean',
                       title=None, label_fct=label_fct, fontsize='large',
                       ax=None, box_side=4)
    plt.suptitle("Acceleration onnxruntime / scikit-learn for DecisionTreeClassifier")
    plt.show()

Configuration
+++++++++++++

.. runpython::
    :rst:
    :warningout: RuntimeWarning
    :showcode:

    from pyquickhelper.pandashelper import df2rst
    import pandas
    name = os.path.join(__WD__, "../../onnx/results/bench_plot_onnxruntime_decision_tree.time.csv")
    df = pandas.read_csv(name)
    print(df2rst(df, number_format=4))

Raw results
+++++++++++

:download:`bench_plot_onnxruntime_decision_tree.csv <../../onnx/results/bench_plot_onnxruntime_decision_tree.perf.csv>`

.. runpython::
    :rst:
    :warningout: RuntimeWarning
    :showcode:
    :toggle: out

    from pyquickhelper.pandashelper import df2rst
    from pymlbenchmark.benchmark.bench_helper import bench_pivot
    import pandas
    name = os.path.join(__WD__, "../../onnx/results/bench_plot_onnxruntime_decision_tree.perf.csv")
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
    name = os.path.join(__WD__, "../../onnx/results/bench_plot_onnxruntime_decision_tree.perf.csv")
    df = pandas.read_csv(name)
    df = df[df['lib'] == 'skl']
    print(df2rst(df, number_format=4))

Benchmark code
++++++++++++++

`bench_plot_onnxruntime_decision_tree.py <https://github.com/sdpython/_benchmarks/blob/master/onnx/bench_plot_onnxruntime_decision_tree.py>`_

.. literalinclude:: ../../onnx/bench_plot_onnxruntime_decision_tree.py
    :language: python
