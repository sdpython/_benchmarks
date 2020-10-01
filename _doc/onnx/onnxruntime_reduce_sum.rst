
.. _l-bench-plot-onnxruntime-reduce-sum:

Benchmark (ONNX) for ReduceSum
==============================

.. contents::
    :local:

.. index:: onnxruntime, Add

The experiment compares the execution time between
:epkg:`numpy` and :epkg:`onnxruntime` for operator
`ReduceSum <https://github.com/onnx/onnx/blob/master/
docs/Operators.md#ReduceSum>`_.

Overview
++++++++

.. plot::

    import matplotlib.pyplot as plt
    import pandas
    from pymlbenchmark.plotting import plot_bench_results

    def custom_label_fct(la):
        la = label_fct(la)
        la = la.replace("edims=(", "edims=(N, ")
        return la

    name = "../../onnx/results/bench_plot_onnxruntime_reduce_sum.perf.csv"
    df = pandas.read_csv(name)
    plot_bench_results(df, row_cols='edims', col_cols='axes',
                       x_value='N', cmp_col_values=('lib', 'npy'),
                       title="Benchmark ReduceSum",
                       label_fct=custom_label_fct)

    plt.suptitle("Acceleration onnxruntime / numpy for ReduceSum")
    plt.show()

Configuration
+++++++++++++

.. runpython::
    :rst:
    :warningout: RuntimeWarning
    :showcode:

    from pyquickhelper.pandashelper import df2rst
    import pandas
    name = os.path.join(__WD__, "../../onnx/results/bench_plot_onnxruntime_reduce_sum.time.csv")
    df = pandas.read_csv(name)
    print(df2rst(df, number_format=4))

Raw results
+++++++++++

:download:`bench_plot_onnxruntime_reduce_sum.csv <../../onnx/results/bench_plot_onnxruntime_reduce_sum.perf.csv>`

.. runpython::
    :rst:
    :warningout: RuntimeWarning
    :showcode:
    :toggle: out

    from pyquickhelper.pandashelper import df2rst
    from pymlbenchmark.benchmark.bench_helper import bench_pivot
    import pandas
    name = os.path.join(__WD__, "../../onnx/results/bench_plot_onnxruntime_reduce_sum.perf.csv")
    df = pandas.read_csv(name)
    piv = bench_pivot(df).reset_index(drop=False)
    piv['speedup'] = piv['npy'] / piv['ort']
    print(df2rst(piv, number_format=4))

Benchmark code
++++++++++++++

`bench_plot_onnxruntime_reduce_sum.py <https://github.com/sdpython/_benchmarks/blob/master/onnx/bench_plot_onnxruntime_reduce_sum.py>`_

.. literalinclude:: ../../onnx/bench_plot_onnxruntime_reduce_sum.py
    :language: python
