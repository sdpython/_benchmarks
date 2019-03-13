
.. _l-bench-plot-module-re2:

re / re2 for date matching
==========================

.. contents::
    :local:

.. index:: re, re2, regular expressions

Overview
++++++++

.. plot::

    import matplotlib.pyplot as plt
    import pandas
    from pymlbenchmark.plotting import plot_bench_xtime

    name = "../../modules/results/bench_re2.csv"
    df = pandas.read_csv(name)

    plot_bench_xtime(df, row_cols='N', col_cols=None,
                     hue_cols=['dim'],
                     cmp_col_values=('test', 're'),
                     x_value='mean', y_value='xtime',
                     parallel=(1., 0.5), title=None,
                     ax=None, box_side=4)
    plt.suptitle("re / re2 for dates")
    plt.show()

Detailed graphs
+++++++++++++++

.. plot::

    import matplotlib.pyplot as plt
    import pandas
    from pymlbenchmark.plotting import plot_bench_results

    name = "../../modules/results/bench_re2.csv"
    df = pandas.read_csv(name)

    plot_bench_results(df, row_cols='N', col_cols=None,
                       hue_cols='max_depth',
                       cmp_col_values=('test', 're'),
                       x_value='dim', y_value='mean',
                       title=None,
                       ax=None, box_side=4)
    plt.suptitle("re / re2 for dates")
    plt.show()

Raw results
+++++++++++

:download:`bench_re2.csv <../../onnx/results/bench_re2.csv>`

.. runpython::
    :rst:
    :warningout: RuntimeWarning
    :showcode:
    :toggle: out

    from pyquickhelper.pandashelper import df2rst
    from pymlbenchmark.benchmark.bench_helper import bench_pivot
    import pandas
    name = os.path.join(__WD__, "../../modules/results/bench_re2.csv")
    df = pandas.read_csv(name)
    piv = bench_pivot(df).reset_index(drop=False)
    piv['speedup'] = piv['re'] / piv['re2']
    print(df2rst(piv, number_format=4))

Benchmark code
++++++++++++++

`bench_plot_re2.py <https://github.com/sdpython/_benchmarks/blob/master/modules/bench_plot_re2.py>`_

.. literalinclude:: ../../modules/bench_plot_re2.py
    :language: python
