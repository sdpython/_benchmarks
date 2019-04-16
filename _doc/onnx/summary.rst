
Prediction time scikit-learn / onnxruntime: summary
===================================================

Summarizing performance is not simple as there is
not simple rule about the gain obtained by using
:epkg:`onnxruntime` over :epkg:`scikit-learn`.
It depends on the number of trees, the depth.

.. contents::
    :local:

Median Prediction Time
++++++++++++++++++++++

.. plot::

    import os
    import pandas
    import matplotlib.pyplot as plt
    from pyquickhelper.pandashelper import df2rst
    from pymlbenchmark.benchmark.bench_helper import bench_pivot

    renamed = {
        'bench_plot_onnxruntime_logreg': 'LogisticRegression',
        'bench_plot_onnxruntime_decision_tree': 'DecisionTreeClassifier',
        'bench_plot_onnxruntime_random_forest': 'RandomForestClassifier',
        'bench_plot_onnxruntime_multinomialnb': 'MultinomialNB',
    }

    folder = "../../onnx/results"
    tests = [os.path.join(folder, n) for n in os.listdir(folder)]
    dfs = [(name, pandas.read_csv(name)) for name in tests \
           if '.time.' not in name and 'unittest' not in name]

    rows = []
    side1 = 'skl'
    side2 = 'ort'
    for name, df in dfs:
        df = bench_pivot(df, value='median').reset_index(drop=False)
        if "dim" in df.columns and 'N' in df.columns and \
            side1 in df.columns and side2 in df.columns and \
            'method' in df.columns:
            key = os.path.splitext(os.path.split(name)[-1])[0]
            subn = df[(df.N == 1) & (df.method == 'predict')]
            for dim in sorted(set(df.dim)):
                subset = subn[subn.dim == dim].sort_values(side2).reset_index(drop=True)
                sel = subset.shape[0] // 2
                row = subset.iloc[sel:sel+1, :].copy()
                row['_name'] = key
                rows.append(row)

    df = pandas.concat(rows, sort=False)
    df.fillna('', inplace=True)
    df['speedup'] = df[side1] / df[side2]
    df = df.sort_values(['_name', 'dim'])
    df['_name'] = df['_name'].apply(lambda x: renamed.get(x, x))
    df = df.drop(['N', 'method', side2], axis=1)
    df = df[['_name', 'speedup']]
    df = df.groupby('_name').median().sort_values('speedup')

    plt.close('all')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
    df.plot(ax=ax, kind='bar', rot=33 , logy=True, label="models")
    ax.set_ylabel('Acceleration\nonnxruntime VS scikit-learn')
    ax.plot([-1, df.shape[0]], [1, 1], '-', label="1x", color='black')
    ax.plot([-1, df.shape[0]], [2, 2], '--', label="2x", color='black')
    ax.plot([-1, df.shape[0]], [10, 10], '--', label="10x", color='black')
    ax.plot([-1, df.shape[0]], [100, 100], '--', label="100x", color='black')
    ax.legend()
    fig.subplots_adjust(bottom=0.25)
    plt.show()

Median Prediction Time per number of features
+++++++++++++++++++++++++++++++++++++++++++++

The following table looks into the results for
every converted model on one observation.
It then takes the median *scikit-learn* time
for each number of features.

.. runpython::
    :rst:
    :warningout: RuntimeWarning
    :showcode:
    :toggle:

    import os
    import pandas
    from pyquickhelper.pandashelper import df2rst
    from pymlbenchmark.benchmark.bench_helper import bench_pivot

    renamed = {
        'bench_plot_onnxruntime_logreg': 'LogisticRegression',
        'bench_plot_onnxruntime_decision_tree': 'DecisionTreeClassifier',
        'bench_plot_onnxruntime_random_forest': 'RandomForestClassifier',
        'bench_plot_onnxruntime_multinomialnb': 'MultinomialNB',
    }

    folder = os.path.join(__WD__, "../../onnx/results")
    tests = [os.path.join(folder, n) for n in os.listdir(folder)]
    dfs = [(name, pandas.read_csv(name)) for name in tests \
           if '.time.' not in name and 'unittest' not in name]

    rows = []
    side1 = 'skl'
    side2 = 'ort'
    for name, df in dfs:
        df = bench_pivot(df, value='median').reset_index(drop=False)
        if "dim" in df.columns and 'N' in df.columns and \
            side1 in df.columns and side2 in df.columns and \
            'method' in df.columns:
            key = os.path.splitext(os.path.split(name)[-1])[0]
            subn = df[(df.N == 1) & (df.method == 'predict')]
            for dim in sorted(set(df.dim)):
                subset = subn[subn.dim == dim].sort_values(side2).reset_index(drop=True)
                sel = subset.shape[0] // 2
                row = subset.iloc[sel:sel+1, :].copy()
                row['_name'] = key
                rows.append(row)

    df = pandas.concat(rows, sort=False)
    df.fillna('', inplace=True)
    df['xtime'] = df[side1] / df[side2]
    df = df.sort_values(['_name', 'dim'])
    df['_name'] = df['_name'].apply(lambda x: renamed.get(x, x))
    df = df.drop(['N', 'method', 'ort'], axis=1)
    cols = ['_name', 'dim', side1, 'xtime']
    cols = cols + [c for c in df.columns if c not in cols]
    df = df[cols]
    print(df2rst(df, number_format=3))

Min Gain Prediction Time
++++++++++++++++++++++++

The following table looks into the results for
every converted model on one observation.
It then takes the minimum gain over *scikit-learn*
for each number of features.

.. runpython::
    :rst:
    :warningout: RuntimeWarning
    :showcode:
    :toggle:

    import os
    import pandas
    from pyquickhelper.pandashelper import df2rst
    from pymlbenchmark.benchmark.bench_helper import bench_pivot

    renamed = {
        'bench_plot_onnxruntime_logreg.perf': 'LogisticRegression',
        'bench_plot_onnxruntime_decision_tree.perf': 'DecisionTreeClassifier',
        'bench_plot_onnxruntime_random_forest.perf': 'RandomForestClassifier',
        'bench_plot_onnxruntime_multinomialnb.perf': 'MultinomialNB',
    }

    folder = os.path.join(__WD__, "../../onnx/results")
    tests = [os.path.join(folder, n) for n in os.listdir(folder)]
    dfs = [(name, pandas.read_csv(name)) for name in tests \
           if '.time.' not in name and 'unittest' not in name]

    rows = []
    side1 = 'skl'
    side2 = 'ort'
    for name, df in dfs:
        df = bench_pivot(df, value='min').reset_index(drop=False)
        if "dim" in df.columns and 'N' in df.columns and \
            side1 in df.columns and side2 in df.columns and \
            'method' in df.columns:
            key = os.path.splitext(os.path.split(name)[-1])[0]
            subn = df[(df.N == 1) & (df.method == 'predict')]
            for dim in sorted(set(df.dim)):
                subset = subn[subn.dim == dim].sort_values(side2).reset_index(drop=True)
                sel = subset.shape[0] // 2
                row = subset.iloc[sel:sel+1, :].copy()
                row['_name'] = key
                rows.append(row)

    df = pandas.concat(rows, sort=False)
    df.fillna('', inplace=True)
    df['xtime'] = df[side1] / df[side2]
    df = df.sort_values(['_name', 'dim'])
    df['_name'] = df['_name'].apply(lambda x: renamed.get(x, x))
    df = df.drop(['N', 'method', 'ort'], axis=1)
    cols = ['_name', 'dim', side1, 'xtime']
    cols = cols + [c for c in df.columns if c not in cols]
    df = df[cols]
    print(df2rst(df, number_format=3))
