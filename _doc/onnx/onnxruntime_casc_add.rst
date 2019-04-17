
.. _l-bench-plot-onnxruntime-cascade-add:

Prediction time numpy / onnxruntime for Addition
================================================

.. contents::
    :local:

.. index:: onnxruntime, Add

The experiment compares the execution time between
:epkg:`numpy` and :epkg:`onnxruntime` for a series
(or cascade) of additions.

.. math::

    y = (((X + M_1) + M_2) + ...) + M_k

*k* is named the number of nodes and the corresponding
*ONNX* graph for *k=2 or 4* looks like a cascade of operators
`Add <https://github.com/onnx/onnx/blob/master/docs/Operators.md#Add>`_.
Both :epkg:`onnxruntime` and :epkg:`numpy` use :epkg:`BLAS` functions.

.. list-table::
    :widths: 5 5

    * - *k=2*
      - *k=4*
    * - .. image:: graph.2.dot.png
      - .. image:: graph.4.dot.png

Overview
++++++++

.. plot::

    import matplotlib.pyplot as plt
    import pandas
    from pymlbenchmark.plotting import plot_bench_results

    name = "../../onnx/results/bench_plot_onnxruntime_casc_add.perf.csv"
    df = pandas.read_csv(name)
    plot_bench_results(df, row_cols='N', col_cols='dim',
                       x_value='nbnode',
                       title="%s\nBenchmark numpy / onnxruntime" % "Cascade Add");

    plt.suptitle("Acceleration onnxruntime / numpy for Add")
    plt.show()

Configuration
+++++++++++++

.. runpython::
    :rst:
    :warningout: RuntimeWarning
    :showcode:

    from pyquickhelper.pandashelper import df2rst
    import pandas
    name = os.path.join(__WD__, "../../onnx/results/bench_plot_onnxruntime_casc_add.time.csv")
    df = pandas.read_csv(name)
    print(df2rst(df, number_format=4))

Raw results
+++++++++++

:download:`bench_plot_onnxruntime_casc_add.csv <../../onnx/results/bench_plot_onnxruntime_casc_add.perf.csv>`

.. runpython::
    :rst:
    :warningout: RuntimeWarning
    :showcode:
    :toggle: out

    from pyquickhelper.pandashelper import df2rst
    from pymlbenchmark.benchmark.bench_helper import bench_pivot
    import pandas
    name = os.path.join(__WD__, "../../onnx/results/bench_plot_onnxruntime_casc_add.perf.csv")
    df = pandas.read_csv(name)
    piv = bench_pivot(df).reset_index(drop=False)
    piv['speedup'] = piv['npy'] / piv['ort']
    print(df2rst(piv, number_format=4))

Benchmark code
++++++++++++++

`bench_plot_onnxruntime_casc_add.py <https://github.com/sdpython/_benchmarks/blob/master/onnx/bench_plot_onnxruntime_casc_add.py>`_

.. literalinclude:: ../../onnx/bench_plot_onnxruntime_casc_add.py
    :language: python
