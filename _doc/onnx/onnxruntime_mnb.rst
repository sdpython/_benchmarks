
.. _l-bench-plot-onnxruntime-multinomialnb:

Prediction time scikit-learn / onnxruntime for MultinomialNB
============================================================

.. index:: onnxruntime, MultinomialNB

.. contents::
    :local:

Code
++++

`bench_plot_onnxruntime_multinomialnb.py <https://github.com/sdpython/_benchmarks/blob/master/onnx/bench_plot_onnxruntime_multinomialnb.py>`_

Overview
++++++++

.. plot::

    import matplotlib.pyplot as plt
    import pandas
    from pymlbenchmark.plotting import plot_bench_xtime

    name = "../../onnx/results/bench_plot_onnxruntime_multinomialnb.perf.csv"
    df = pandas.read_csv(name)

    plot_bench_xtime(df, row_cols='N', col_cols='method',
                     hue_cols='fit_prior',
                     cmp_col_values=('lib', 'skl'),
                     x_value='mean', y_value='xtime',
                     parallel=(1., 0.5), title=None,
                     ax=None, box_side=4)
    plt.suptitle("Acceleration onnxruntime / scikit-learn for MultinomialNB")
    plt.show()

.. plot::

    import matplotlib.pyplot as plt
    import pandas
    from pymlbenchmark.plotting import plot_bench_xtime

    name = "../../onnx/results/bench_plot_onnxruntime_multinomialnb.perf.csv"
    df = pandas.read_csv(name)

    plot_bench_xtime(df, row_cols='N', col_cols='method',
                     hue_cols='alpha',
                     cmp_col_values=('lib', 'skl'),
                     x_value='mean', y_value='xtime',
                     parallel=(1., 0.5), title=None,
                     ax=None, box_side=4)
    plt.suptitle("Acceleration onnxruntime / scikit-learn for MultinomialNB")
    plt.show()

Detailed graphs
+++++++++++++++

.. plot::

    import matplotlib.pyplot as plt
    import pandas
    from pymlbenchmark.plotting import plot_bench_results

    name = "../../onnx/results/bench_plot_onnxruntime_multinomialnb.perf.csv"
    df = pandas.read_csv(name)

    plot_bench_results(df, row_cols=['N', 'alpha'], col_cols='method',
                       hue_cols='fit_prior',
                       cmp_col_values=('lib', 'skl'),
                       x_value='dim', y_value='mean',
                       title=None,
                       ax=None, box_side=4)
    plt.suptitle("Acceleration onnxruntime / scikit-learn for MultinomialNB")
    plt.show()

Configuration
+++++++++++++

.. runpython::
    :rst:
    :warningout: RuntimeWarning
    :showcode:

    from pyquickhelper.pandashelper import df2rst
    import pandas
    name = os.path.join(__WD__, "../../onnx/results/bench_plot_onnxruntime_multinomialnb.time.csv")
    df = pandas.read_csv(name)
    print(df2rst(df, number_format=4))

Raw results
+++++++++++

:download:`bench_plot_onnxruntime_multinomialnb.csv <../../onnx/results/bench_plot_onnxruntime_multinomialnb.perf.csv>`

.. runpython::
    :rst:
    :warningout: RuntimeWarning
    :showcode:
    :toggle: out

    from pyquickhelper.pandashelper import df2rst
    from pymlbenchmark.benchmark.bench_helper import bench_pivot
    import pandas
    name = os.path.join(__WD__, "../../onnx/results/bench_plot_onnxruntime_multinomialnb.perf.csv")
    df = pandas.read_csv(name)
    piv = bench_pivot(df).reset_index(drop=False)
    piv['speedup'] = piv['skl'] / piv['ort']
    print(df2rst(piv, number_format=4))

Benchmark code
++++++++++++++

.. literalinclude:: ../../onnx/bench_plot_onnxruntime_multinomialnb.py
    :language: python
