
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
    from pyquickhelper.pandashelper import df2rst
    import matplotlib.pyplot as plt

    renamed = {
        'bench_plot_onnxruntime_logreg': 'Logistic Regression',
        'bench_plot_onnxruntime_decision_tree': 'Decision Tree',
        'bench_plot_onnxruntime_random_forest': 'Random Forest',
    }

    folder = "../../onnx/results"
    tests = [os.path.join(folder, n) for n in os.listdir(folder)]
    dfs = [(name, pandas.read_csv(name)) for name in tests if '.time.' not in name]

    rows = []
    for name, df in dfs:
        if "nfeat" in df.columns and 'n_obs' in df.columns and \
            'time_skl' in df.columns and 'time_ort' in df.columns and \
            'method' in df.columns:
            key = os.path.splitext(os.path.split(name)[-1])[0]
            subn = df[(df.n_obs == 1) & (df.method == 'predict')]
            for nfeat in sorted(set(df.nfeat)):
                subset = subn[subn.nfeat == nfeat].sort_values('time_skl').reset_index(drop=True)
                sel = subset.shape[0] // 2
                row = subset.iloc[sel:sel+1, :].copy()
                row['_name'] = key
                rows.append(row)

    df = pandas.concat(rows, sort=False)
    df.fillna('', inplace=True)
    df['speedup'] = df['time_skl'] / df['time_ort']
    df = df.sort_values(['_name', 'nfeat'])
    df['_name'] = df['_name'].apply(lambda x: renamed.get(x, x))
    df = df.drop(['n_obs', 'method', 'time_ort'], axis=1)
    df = df[['_name', 'speedup']]
    df = df.groupby('_name').median().sort_values('speedup')

    plt.close('all')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
    df.plot(ax=ax, kind='bar', rot=33 , logy=True, label="models")
    ax.set_ylabel('speed up\nonnxruntime VS scikit-learn')
    ax.plot([-1, df.shape[0]], [1, 1], '-', label="1x", color='black')
    ax.plot([-1, df.shape[0]], [2, 2], '--', label="2x", color='black')
    ax.plot([-1, df.shape[0]], [10, 10], '--', label="10x", color='black')
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

    renamed = {
        'bench_plot_onnxruntime_logreg': 'Logistic Regression',
        'bench_plot_onnxruntime_decision_tree': 'Decision Tree',
        'bench_plot_onnxruntime_random_forest': 'Random Forest',
    }

    folder = os.path.join(__WD__, "../../onnx/results")
    tests = [os.path.join(folder, n) for n in os.listdir(folder)]
    dfs = [(name, pandas.read_csv(name)) for name in tests if '.time.' not in name]

    rows = []
    for name, df in dfs:
        if "nfeat" in df.columns and 'n_obs' in df.columns and \
            'time_skl' in df.columns and 'time_ort' in df.columns and \
            'method' in df.columns:
            key = os.path.splitext(os.path.split(name)[-1])[0]
            subn = df[(df.n_obs == 1) & (df.method == 'predict')]
            for nfeat in sorted(set(df.nfeat)):
                subset = subn[subn.nfeat == nfeat].sort_values('time_skl').reset_index(drop=True)
                sel = subset.shape[0] // 2
                row = subset.iloc[sel:sel+1, :].copy()
                row['_name'] = key
                rows.append(row)

    df = pandas.concat(rows, sort=False)
    df.fillna('', inplace=True)
    df['speedup'] = df['time_skl'] / df['time_ort']
    df = df.sort_values(['_name', 'nfeat'])
    df['_name'] = df['_name'].apply(lambda x: renamed.get(x, x))
    df = df.drop(['n_obs', 'method', 'time_ort'], axis=1)
    cols = ['_name', 'nfeat', 'time_skl', 'speedup']
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

    renamed = {
        'bench_plot_onnxruntime_logreg': 'Logistic Regression',
        'bench_plot_onnxruntime_decision_tree': 'Decision Tree',
        'bench_plot_onnxruntime_random_forest': 'Random Forest',
    }

    folder = os.path.join(__WD__, "../../onnx/results")
    tests = [os.path.join(folder, n) for n in os.listdir(folder)]
    dfs = [(name, pandas.read_csv(name)) for name in tests if '.time.' not in name]

    rows = []
    for name, df in dfs:
        if "nfeat" in df.columns and 'n_obs' in df.columns and \
            'time_skl' in df.columns and 'time_ort' in df.columns and \
            'method' in df.columns:
            key = os.path.splitext(os.path.split(name)[-1])[0]
            subn = df[(df.n_obs == 1) & (df.method == 'predict')].copy()
            subn['speedup'] = subn['time_skl'] / subn['time_ort']
            for nfeat in sorted(set(df.nfeat)):
                subset = subn[subn.nfeat == nfeat].sort_values('speedup').reset_index(drop=True)
                sel = subset.shape[0] // 2
                row = subset.iloc[:1, :].copy()
                row['_name'] = key
                rows.append(row)

    df = pandas.concat(rows, sort=False)
    df.fillna('', inplace=True)
    df = df.sort_values(['_name', 'nfeat'])
    df['_name'] = df['_name'].apply(lambda x: renamed.get(x, x))
    df = df.drop(['n_obs', 'method', 'time_ort'], axis=1)
    cols = ['_name', 'nfeat', 'time_skl', 'speedup']
    cols = cols + [c for c in df.columns if c not in cols]
    df = df[cols]
    print(df2rst(df, number_format=3))
