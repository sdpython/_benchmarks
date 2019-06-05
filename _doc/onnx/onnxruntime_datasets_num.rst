
.. _l-bench-plot-onnxruntime-datasets-nul:

Prediction time scikit-learn / onnxruntime for common datasets
==============================================================

.. contents::
    :local:

.. index:: onnxruntime, datasets, breast cancer, digits

Overview
++++++++

The following graph plots the ratio between :epkg:`onnxruntime`
and :epkg:`scikit-learn`. It looks into multiple models for
a couple of datasets :
`breast cancer <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html>`_,
`digits <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html>`_.
It computes the prediction time for the following models:

* *ADA*: ``AdaBoostClassifier()``
* *BNB*: ``BernoulliNB()``
* *DT*: ``DecisionTreeClassifier(max_depth=4)``
* *GBT*: ``GradientBoostingClassifier(max_depth=4, n_estimators=10)``
* *KNN*: ``KNeighborsClassifier()``
* *LR*: ``LogisticRegression(solver="liblinear", penalty="l2")``
* *MLP*: ``MLPClassifier()``
* *MNB*: ``MultinomialNB()``
* *NuSVC*: ``NuSVC(probability=True)``
* *RF*: ``RandomForestClassifier(max_depth=4, n_estimators=10)``
* *SVC*: ``SVC(probability=True)``

.. plot::

    import matplotlib.pyplot as plt
    import pandas
    from pymlbenchmark.plotting import plot_bench_xtime

    name = "../../onnx/results/bench_plot_datasets_num.perf.csv"
    df = pandas.read_csv(name)
    plot_bench_xtime(df, col_cols='dataset',
                     hue_cols='model',
                     title="Numerical datasets\nBenchmark scikit-learn / onnxruntime")
    plt.show()

Graph X = number of observations to predict
+++++++++++++++++++++++++++++++++++++++++++

.. plot::

    import matplotlib.pyplot as plt
    import pandas
    from pymlbenchmark.plotting import plot_bench_results

    name = "../../onnx/results/bench_plot_datasets_num.perf.csv"
    df = pandas.read_csv(name)
    plot_bench_results(df, row_cols='model', col_cols='dataset',
                       x_value='N',
                       title="Numerical datasets\nBenchmark scikit-learn / onnxruntime")
    plt.show()

Graph of differences between scikit-learn and onnxruntime
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. plot::

    import matplotlib.pyplot as plt
    import pandas
    from pymlbenchmark.plotting import plot_bench_results

    name = "../../onnx/results/bench_plot_datasets_num.perf.csv"
    df = pandas.read_csv(name)
    plot_bench_results(df, row_cols='model', col_cols='dataset',
                       x_value='N', y_value='diff',
                       err_value=('lower_diff', 'upper_diff'),
                       title="Numerical datasets\Absolute difference scikit-learn / onnxruntime")
    plt.show()

Configuration
+++++++++++++

.. runpython::
    :rst:
    :warningout: RuntimeWarning
    :showcode:

    from pyquickhelper.pandashelper import df2rst
    import pandas
    name = os.path.join(__WD__, "../../onnx/results/bench_plot_datasets_num.time.csv")
    df = pandas.read_csv(name)
    print(df2rst(df, number_format=4))

Raw results
+++++++++++

:download:`bench_plot_datasets_num.csv <../../onnx/results/bench_plot_datasets_num.perf.csv>`

.. runpython::
    :rst:
    :warningout: RuntimeWarning
    :showcode:
    :toggle: out

    from pyquickhelper.pandashelper import df2rst
    from pymlbenchmark.benchmark.bench_helper import bench_pivot
    import pandas
    name = os.path.join(__WD__, "../../onnx/results/bench_plot_datasets_num.perf.csv")
    df = pandas.read_csv(name)
    print(df2rst(df, number_format=4))

Benchmark code
++++++++++++++

`bench_plot_datasets_num.py <https://github.com/sdpython/_benchmarks/blob/master/onnx/bench_plot_datasets_num.py>`_

.. literalinclude:: ../../onnx/bench_plot_datasets_num.py
    :language: python
