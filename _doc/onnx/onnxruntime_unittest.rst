
.. _l-bench-plot-onnxruntime-unittest:

Benchmark (ONNX) for sklearn-onnx unit tests
============================================

.. contents::
    :local:

.. index:: onnxruntime, unit tests

Overview
++++++++

The following graph plots the ratio between :epkg:`onnxruntime`
and :epkg:`scikit-learn`. The lower, the better for :epkg:`onnxruntime`.
Each test is mapped to a unit test in :epkg:`sklearn-onnx`
in folder `tests <https://github.com/onnx/sklearn-onnx/tree/master/tests>`_.

.. plot::

    import matplotlib.pyplot as plt
    import pandas

    name = "../../onnx/results/bench_plot_skl2onnx_unittest.perf.csv"
    df = pandas.read_csv(name)
    df = df[df['stderr'].isnull() & ~df.ratio.isnull()].sort_values("ratio").copy()
    df['model'] = df['_model'].apply(lambda s: s.replace("Sklearn", ""))

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    df.plot(x="original_time", y="ratio", ax=ax, logx=True, logy=True, kind="scatter")
    xmin, xmax = df.original_time.min(), df.original_time.max()
    ax.plot([xmin, xmax], [1, 1], "--", label="1x")
    ax.plot([xmin, xmax], [2, 2], "--", label="2x slower")
    ax.plot([xmin, xmax], [0.5, 0.5], "--", label="2x faster")
    ax.set_title("Ratio onnxruntime / scikit-learn\nLower is better")
    ax.set_xlabel("execution time with scikit-learn (seconds)")
    ax.set_ylabel("Ratio onnxruntime / scikit-learn\nLower is better.")
    ax.legend()
    fig.tight_layout()

    plt.show()

Ratio by model
++++++++++++++

.. plot::

    import matplotlib.pyplot as plt
    import pandas

    name = "../../onnx/results/bench_plot_skl2onnx_unittest.perf.csv"
    df = pandas.read_csv(name)
    df = df[df['stderr'].isnull() & ~df.ratio.isnull()].sort_values("ratio").copy()
    df['model'] = df['_model'].apply(lambda s: s.replace("Sklearn", ""))

    fig, ax = plt.subplots(1, 1, figsize=(10, 80))
    df.plot.barh(x="model", y="ratio", ax=ax, logx=True)
    ymin, ymax = ax.get_ylim()
    ax.plot([0.5, 0.5], [ymin, ymax], '--', label="2x faster")
    ax.plot([1, 1], [ymin, ymax], '-', label="1x")
    ax.plot([2, 2], [ymin, ymax], '--', label="2x slower")
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(8)
    ax.legend(loc='upper left')
    ax.grid()
    ax.set_title("Ratio onnxruntime / scikit-learn\nLower is better")
    fig.tight_layout()
    plt.show()

Configuration
+++++++++++++

.. runpython::
    :rst:
    :warningout: RuntimeWarning
    :showcode:

    from pyquickhelper.pandashelper import df2rst
    import pandas
    name = os.path.join(__WD__, "../../onnx/results/bench_plot_skl2onnx_unittest.time.csv")
    df = pandas.read_csv(name)
    print(df2rst(df, number_format=4))

Errors
++++++

.. runpython::
    :rst:
    :warningout: RuntimeWarning
    :showcode:

    from pyquickhelper.pandashelper import df2rst
    import pandas
    name = os.path.join(__WD__, "../../onnx/results/bench_plot_skl2onnx_unittest.perf.csv")
    df = pandas.read_csv(name)
    err = df[~df['stderr'].isnull()]
    err = err[["_model", "stderr"]]
    print(df2rst(err))

Raw results
+++++++++++

:download:`bench_plot_skl2onnx_unittest.csv <../../onnx/results/bench_plot_skl2onnx_unittest.perf.csv>`

.. runpython::
    :rst:
    :warningout: RuntimeWarning
    :showcode:
    :toggle: out

    from pyquickhelper.pandashelper import df2rst
    from pymlbenchmark.benchmark.bench_helper import bench_pivot
    import pandas
    name = os.path.join(__WD__, "../../onnx/results/bench_plot_skl2onnx_unittest.perf.csv")
    df = pandas.read_csv(name)
    df = df[df['stderr'].isnull() & ~df.ratio.isnull()].sort_values("ratio").copy()
    df['model'] = df['_model'].apply(lambda s: s.replace("Sklearn", ""))
    piv = df[["model", "original_time", "original_std", "onnxrt_time", "onnxrt_std", "ratio"]]
    piv.columns = ["model", "sklearn", "skl dev", "onnxruntime", "ort dev", "ratio"]
    print(df2rst(piv, number_format=4))

Benchmark code
++++++++++++++

`bench_plot_skl2onnx_unittest.py <https://github.com/sdpython/_benchmarks/blob/master/onnx/bench_plot_skl2onnx_unittest.py>`_

.. literalinclude:: ../../onnx/bench_plot_skl2onnx_unittest.py
    :language: python
