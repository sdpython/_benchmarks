
.. _l-bench-plot-onnxruntime-datasets-nul-reg-knn:

Benchmark (ONNX) for common datasets (regression) with k-NN
===========================================================

.. contents::
    :local:

.. index:: onnxruntime, datasets, boston, diabetes

Overview
++++++++

The following graph plots the ratio between :epkg:`onnxruntime`
and :epkg:`scikit-learn`. It looks into multiple models for
a couple of datasets :
`boston <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html>`_,
`diabetes <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html>`_.
It computes the prediction time for the following models:

* *ADA*: ``AdaBoostRegressor()``
* *DT*: ``DecisionTreeRegressor(max_depth=6)``
* *GBT*: ``GradientBoostingRegressor(max_depth=6, n_estimators=100)``
* *KNN*: ``KNeighborsRegressor()``
* *KNN-cdist*: ``KNeighborsRegressor()``, the conversion to ONNX
  is run with option ``{'optim': 'cdist'}`` to use a specific operator
  to compute pairwise distances
* *LGB*: ``LGBMRegressor(max_depth=6, n_estimators=100)``
* *LR*: ``LinearRegression(solver="liblinear", penalty="l2")``
* *MLP*: ``MLPRegressor()``
* *NuSVR*: ``NuSVC(probability=True)``
* *RF*: ``RandomForestRegressor(max_depth=6, n_estimators=100)``
* *SVR*: ``SVC(probability=True)``
* *XGB*: ``XGBRegressor(max_depth=6, n_estimators=100)``

The predictor follows a `StandardScaler
<https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html>`_ (or a
`MinMaxScaler <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html>`_
if the model is a Naive Bayes one)
in a pipeline if
``norm=True`` or is the only object is ``norm=False``.
The pipeline looks like
``make_pipeline(StandardScaler(), estimator())``.
Three runtimes are tested:

* `skl`: :epkg:`scikit-learn`,
* `ort`: :epkg:`onnxruntime`,
* `pyrt`: :epkg:`mlprodict`, it relies on :epkg:`numpy`
  for most of the operators except trees and svm which
  use a modified version of the C++ code embedded in
  :epkg:`onnxruntime`.

.. plot::

    import matplotlib.pyplot as plt
    import pandas
    from pymlbenchmark.plotting import plot_bench_xtime

    name = "../../onnx/results/bench_plot_datasets_num_reg_knn.perf.csv"
    df = pandas.read_csv(name)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    plot_bench_xtime(df[~df.norm], col_cols='dataset',
                     hue_cols='model', fontsize=24,
                     title="Numerical datasets - norm=False\nBenchmark scikit-learn / onnxruntime",
                     ax=ax)
    fig.show()

Graph X = number of observations to predict
+++++++++++++++++++++++++++++++++++++++++++

.. plot::

    import matplotlib.pyplot as plt
    import pandas
    from pymlbenchmark.plotting import plot_bench_results

    name = "../../onnx/results/bench_plot_datasets_num_reg_knn.perf.csv"
    df = pandas.read_csv(name)
    plot_bench_results(df, row_cols='model', col_cols=('dataset', 'norm'),
                       x_value='N', fontsize=24,
                       title="Numerical datasets\nBenchmark scikit-learn / onnxruntime")
    plt.show()

Graph computing time per observations
+++++++++++++++++++++++++++++++++++++

The following graph shows the computing cost per
observations depending on the batch size.
:epkg:`scikit-learn` is clearly optimized for batch predictions
(= training).

.. plot::

    import matplotlib.pyplot as plt
    import pandas
    from pymlbenchmark.plotting import plot_bench_results

    name = "../../onnx/results/bench_plot_datasets_num_reg_knn.perf.csv"
    df = pandas.read_csv(name)
    for c in "min,max,mean,lower,upper,median".split(','):
        df[c] /= df['N']
    plot_bench_results(df, row_cols='model', col_cols=('dataset', 'norm'),
                       x_value='N', 24,
                       title="Numerical datasets\nBenchmark scikit-learn / onnxruntime")
    plt.show()

Graph of differences between scikit-learn and onnxruntime
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. plot::

    import matplotlib.pyplot as plt
    import pandas
    from pymlbenchmark.plotting import plot_bench_results

    name = "../../onnx/results/bench_plot_datasets_num_reg_knn.perf.csv"
    df = pandas.read_csv(name)
    plot_bench_results(df, row_cols='model', col_cols=('dataset', 'norm'),
                       x_value='N', y_value='diff_ort', 24,
                       err_value=('lower_diff_ort', 'upper_diff_ort'),
                       title="Numerical datasets\Absolute difference scikit-learn / onnxruntime")
    plt.show()

Graph of differences between scikit-learn and python runtime
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. plot::

    import matplotlib.pyplot as plt
    import pandas
    from pymlbenchmark.plotting import plot_bench_results

    name = "../../onnx/results/bench_plot_datasets_num_reg_knn.perf.csv"
    df = pandas.read_csv(name)
    plot_bench_results(df, row_cols='model', col_cols=('dataset', 'norm'),
                       x_value='N', y_value='diff_pyrt', fontsize=24,
                       err_value=('lower_diff_pyrt', 'upper_diff_pyrt'),
                       title="Numerical datasets\Absolute difference scikit-learn / python runtime")
    plt.show()

Configuration
+++++++++++++

.. runpython::
    :rst:
    :warningout: RuntimeWarning
    :showcode:

    from pyquickhelper.pandashelper import df2rst
    import pandas
    name = os.path.join(__WD__, "../../onnx/results/bench_plot_datasets_num_reg_knn.time.csv")
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
    name = os.path.join(__WD__, "../../onnx/results/bench_plot_datasets_num_reg_knn.perf.csv")
    df = pandas.read_csv(name)
    print(df2rst(df, number_format=4))

Benchmark code
++++++++++++++

`bench_plot_datasets_num.py <https://github.com/sdpython/_benchmarks/blob/master/onnx/bench_plot_datasets_num_reg_knn.py>`_

.. literalinclude:: ../../onnx/bench_plot_datasets_num_reg_knn.py
    :language: python
