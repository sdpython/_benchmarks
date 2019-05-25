
.. _l-gridsearch-cache:

Caching algorithm for a GridSearchCV
====================================

.. index:: grid search, cache, scikit-learn, joblib

.. contents::
    :local:

Ideas
+++++

The goal is to measure the impact of using a cache
while optimizing a pipeline:

::

    [
        ('scale', MinMaxScaler()),
        ('pca', PCA(2)),
        ('poly', PolynomialFeatures()),
        ('bins', KBinsDiscretizer()),
        ('lr', LogisticRegression(solver='liblinear'))
    ]

With the following parameters:

::

    params_grid = {
        'scale__feature_range': [(0, 1), (-1, 1)],
        'pca__n_components': [2, 4],
        'poly__degree': [2, 3],
        'bins__n_bins': [5],
        'bins__encode': ["onehot-dense", "ordinal"],
        'lr__penalty': ['l1', 'l2'],
    }

It looks into different ways to speed up the optimization
by caching. One option is not implemented in
:epkg:`scikit-learn`: `PipelineCache
<https://github.com/sdpython/mlinsights/blob/master/mlinsights/mlbatch/pipeline_cache.py>`_,
it implements a cache in memory as opposed of :epkg:`joblib`
which stores everything on disk. This implementation is faster
when the training runs with one process, :epkg:`joblib` does a better
job if the number of jobs and processes is higher even if it may
store a huge load of data.

Graphs
++++++

.. plot::

    import matplotlib.pyplot as plt
    import pandas
    from pymlbenchmark.plotting plot_bench_results
    name = "../../scikit-learn/results/bench_plot_gridsearch_cache.csv"
    df = pandas.read_csv(name)
    plt.close('all')

    plot_bench_results(df, row_cols=['N'],
                       col_cols=['n_jobs'], x_value='dim',
                       hue_cols=['test'],
                       cmp_col_values='test',
                       title="GridSearchCV\nBenchmark caching strategies")
    plt.show()

.. plot::

    import matplotlib.pyplot as plt
    import pandas
    from pymlbenchmark.plotting plot_bench_xtime
    name = "../../scikit-learn/results/bench_plot_gridsearch_cache.csv"
    df = pandas.read_csv(name)
    plt.close('all')

    plot_bench_xtime(df, row_cols=['n_jobs'],
                       hue_cols=['N'], x_value='mean',
                       cmp_col_values='test',
                       title="GridSearchCV\nBenchmark caching strategies");
    plt.show()

Machine used to run the test
++++++++++++++++++++++++++++

.. runpython::
    :rst:
    :warningout: RuntimeWarning
    :showcode:

    from pyquickhelper.pandashelper import df2rst
    import pandas
    name = os.path.join(__WD__, "../../scikit-learn/results/bench_plot_gridsearch_cache.time.csv")
    df = pandas.read_csv(name)
    print(df2rst(df, number_format=4))

Raw results
+++++++++++

:download:`bench_polynomial_features.csv <../../scikit-learn/results/bench_plot_gridsearch_cache.csv>`

.. runpython::
    :rst:
    :warningout: RuntimeWarning
    :showcode:

    from pyquickhelper.pandashelper import df2rst
    import pandas
    name = os.path.join(__WD__, "../../scikit-learn/results/bench_plot_gridsearch_cache.csv")
    df = pandas.read_csv(name)
    df['speedup'] = df['time_0_20_2'] / df['time_current']
    print(df2rst(df, number_format=4))

Benchmark code
++++++++++++++

.. literalinclude:: ../../scikit-learn/bench_plot_gridsearch_cache.py
    :language: python
