
.. _l-bench-plot-onnxruntime-multinomialnb:

Prediction time scikit-learn / onnxruntime: MultinomialNB
=========================================================

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
    name = "../../onnx/results/bench_plot_onnxruntime_multinomialnb.csv"
    df = pandas.read_csv(name)
    df['speedup'] = df['time_skl'] / df['time_ort']
    plt.close('all')
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    # method
    for color, method in zip('rgbymc', sorted(set(df.method))):
        subdf = df[df.method == method]
        subdf.plot(x="time_skl", y="speedup", logx=True, logy=True,
                   kind="scatter", ax=ax[0, 0], label=method,
                   c=color)
    ax[0, 0].set_xlabel("Time(s) of scikit-learn\n.")
    ax[0, 0].plot([df.time_skl.min(), df.time_skl.max()], [1, 1],
               "-", c="black", label="1x")
    ax[0, 0].plot([df.time_skl.min(), df.time_skl.max()], [2, 2],
               "--", c="black", label="2x")
    ax[0, 0].plot([df.time_skl.min(), df.time_skl.max()], [10, 10],
               "--", c="black", label="10x")
    ax[0, 0].legend()

    # alpha
    for color, alpha in zip('rgbymc', sorted(set(df.alpha))):
        subdf = df[df.alpha == alpha]
        subdf.plot(x="time_skl", y="speedup", logx=True, logy=True,
                   kind="scatter", ax=ax[0, 1], label="alpha=%f" % alpha,
                   c=color)
    ax[0, 1].set_title("Acceleration / original time")
    ax[0, 1].plot([df.time_skl.min(), df.time_skl.max()], [1, 1],
               "-", c="black", label="1x")
    ax[0, 1].plot([df.time_skl.min(), df.time_skl.max()], [2, 2],
               "--", c="black", label="2x")
    ax[0, 1].plot([df.time_skl.min(), df.time_skl.max()], [10, 10],
               "--", c="black", label="10x")
    ax[0, 1].legend()

    # fit_prior
    for color, fit_prior in zip('rgbymc', sorted(set(df.fit_prior))):
        subdf = df[df.fit_prior == fit_prior]
        subdf.plot(x="time_skl", y="speedup", logx=True, logy=True,
                   kind="scatter", ax=ax[1, 1], label="fit_prior=%d" % fit_prior,
                   c=color)
    ax[1, 1].set_xlabel("Time(s) of scikit-learn\n.")
    ax[1, 1].set_ylabel("Speed up compare to scikit-learn")
    ax[1, 1].plot([df.time_skl.min(), df.time_skl.max()], [1, 1],
               "-", c="black", label="1x")
    ax[1, 1].plot([df.time_skl.min(), df.time_skl.max()], [2, 2],
               "--", c="black", label="2x")
    ax[1, 1].plot([df.time_skl.min(), df.time_skl.max()], [10, 10],
               "--", c="black", label="10x")
    ax[1, 1].legend()

    # features
    for color, nfeat in zip('rgbymc', sorted(set(df.nfeat))):
        subdf = df[df.nfeat == nfeat]
        subdf.plot(x="time_skl", y="speedup", logx=True, logy=True,
                   kind="scatter", ax=ax[1, 0], label="nfeat=%d" % nfeat,
                   c=color)
    ax[1, 0].set_ylabel("Speed up compare to scikit-learn")
    ax[1, 0].plot([df.time_skl.min(), df.time_skl.max()], [1, 1],
               "-", c="black", label="1x")
    ax[1, 0].plot([df.time_skl.min(), df.time_skl.max()], [2, 2],
               "--", c="black", label="2x")
    ax[1, 0].plot([df.time_skl.min(), df.time_skl.max()], [10, 10],
               "--", c="black", label="10x")
    ax[1, 0].legend()

    plt.suptitle("Acceleration onnxruntime / scikit-learn for RandomForest")
    plt.show()

:epkg:`onnxruntime` is always faster in that particular scenario.

Raw results
+++++++++++

:download:`bench_plot_onnxruntime_multinomialnb.csv <../../onnx/results/bench_plot_onnxruntime_multinomialnb.csv>`

.. runpython::
    :rst:
    :warningout: RuntimeWarning
    :showcode:

    from pyquickhelper.pandashelper import df2rst
    import pandas
    name = os.path.join(__WD__, "../../onnx/results/bench_plot_onnxruntime_multinomialnb.csv")
    df = pandas.read_csv(name)
    df['speedup'] = df['time_skl'] / df['time_ort']
    print(df2rst(df, number_format=4))

Detailed graphs
+++++++++++++++

.. plot::

    import matplotlib.pyplot as plt
    import pandas
    name = "../../onnx/results/bench_plot_onnxruntime_multinomialnb.csv"
    df = pandas.read_csv(name)
    plt.close('all')

    nrows = max(len(set(df.fit_prior)) * len(set(df.n_obs)), 2)
    ncols = max(len(set(df.method)), 2)
    fig, ax = plt.subplots(nrows, ncols,
                           figsize=(ncols * 4, nrows * 4))
    pos = 0
    row = 0
    for n_obs in sorted(set(df.n_obs)):
        for fit_prior in sorted(set(df.fit_prior)):
            pos = 0
            for method in sorted(set(df.method)):
                a = ax[row, pos]
                if row == ax.shape[0] - 1:
                    a.set_xlabel("N features", fontsize='x-small')
                if pos == 0:
                    a.set_ylabel("Time (s) n_obs={}\nfit_prior={}".format(n_obs, fit_prior),
                                 fontsize='x-small')

                for color, alpha in zip('rgbcym', sorted(set(df.alpha))):
                    subset = df[(df.method == method) & (df.n_obs == n_obs) &
                                (df.fit_prior == fit_prior) & (df.alpha == alpha)]
                    if subset.shape[0] == 0:
                        continue
                    subset = subset.sort_values("nfeat")
                    label = "skl a=%f" % alpha
                    subset.plot(x="nfeat", y="time_skl", label=label, ax=a,
                                logx=True, logy=True, c=color, style='--')
                    label = "ort a=%f" % alpha
                    subset.plot(x="nfeat", y="time_ort", label=label, ax=a,
                                logx=True, logy=True, c=color)

                a.legend(loc=0, fontsize='x-small')
                if row == 0:
                    a.set_title("method={}".format(method), fontsize='x-small')
                pos += 1
            row += 1

    plt.suptitle(
        "Benchmark for MultinomialNB sklearn/onnxruntime", fontsize=16)

Benchmark code
++++++++++++++

.. literalinclude:: ../../onnx/bench_plot_onnxruntime_multinomialnb.py
    :language: python

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
