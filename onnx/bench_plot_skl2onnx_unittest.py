# coding: utf-8
"""
Benchmark of :epkg:`onnxruntime` on all unit tests from
:epkg:`skl2onnx`.
"""
# Authors: Xavier Dupré (benchmark)
# License: MIT
import matplotlib
matplotlib.use('Agg')

import os
import unittest
import warnings
import contextlib
from time import perf_counter as time
from io import StringIO
import numpy
import pandas
import matplotlib.pyplot as plt
from sklearn.utils.testing import ignore_warnings
from sklearn.utils.extmath import softmax
from pyquickhelper.loghelper import run_cmd, sys_path_append
from pymlbenchmark.context import machine_information


def run_all_tests(location, folder=None, verbose=True):
    """
    Runs all unit tests or unit tests specific to one library.
    The tests produce a series of files dumped into ``folder``
    which can be later used to tests a backend (or a runtime).
    """
    if folder is None:
        raise ValueError("folder cannot be None")
    os.environ["ONNXTESTDUMP"] = folder
    os.environ["ONNXTESTDUMPERROR"] = "1"
    os.environ["ONNXTESTBENCHMARK"] = "1"

    if verbose:
        print("[benchmark] look into '{0}'".format(location))
        print("[benchmark] dump into '{0}'".format(folder))

    subs = [location]
    loader = unittest.TestLoader()
    suites = []

    for sub in subs:
        fold = os.path.join(this, sub)
        if not os.path.exists(fold):
            raise FileNotFoundError("Unable to find '{0}'".format(fold))

        with sys_path_append(fold):
            names = [_ for _ in os.listdir(fold) if _.startswith("test")]
            for name in names:
                if "test_utils" in name:
                    continue
                if "dump" in name.lower():
                    continue
                name = os.path.splitext(name)[0]
                ts = loader.loadTestsFromName(name)
                suites.append(ts)

    with warnings.catch_warnings():
        warnings.filterwarnings(category=DeprecationWarning, action="ignore")
        warnings.filterwarnings(category=FutureWarning, action="ignore")
        st = StringIO()
        runner = unittest.TextTestRunner(st, verbosity=0)
        name = ""
        for tsi, ts in enumerate(suites):
            for k in ts:
                try:
                    for t in k:
                        name = t.__class__.__name__
                        break
                except TypeError as e:
                    warnings.warn(
                        "[ERROR] Unable to run test '{}' - {}.".format(ts, str(e).replace("\n", " ")))
            if verbose:
                print("[benchmark] {}/{}: '{}'".format(tsi + 1, len(suites), name))
            with contextlib.redirect_stderr(st):
                with contextlib.redirect_stdout(st):
                    runner.run(ts)

    from test_utils.tests_helper import make_report_backend
    df = make_report_backend(folder, as_df=True)
    return df


#########################
# Clones skl2onnx
# +++++++++++++++

this = os.path.abspath(os.path.dirname(__file__))
skl = os.path.join(this, "sklearn-onnx")
if os.path.exists(skl):
    pth = skl
    cmd = "git pull"
else:
    pth = None
    cmd = "git clone https://github.com/onnx/sklearn-onnx.git"
run_cmd(cmd, wait=True, change_path=pth, fLOG=print)

#########################
# Runs the benchmark
# ++++++++++++++++++

folder = os.path.join(this, 'onnxruntime-skl2onnx')
location = os.path.join(this, 'sklearn-onnx', "tests")
filename = os.path.splitext(os.path.split(__file__)[-1])[0]
full_filename = filename + ".perf.csv"
if not os.path.exists(full_filename):
    df = run_all_tests(location, folder, verbose=True)
    print("[benchmark] saves into '{}'.".format(full_filename))
    df.to_csv(full_filename, index=False)
else:
    print("[benchmark] restores from '{}'.".format(full_filename))
    df = pandas.read_csv(full_filename)
print(df.head())

#########################
# Extracts information about the machine used
# +++++++++++++++++++++++++++++++++++++++++++

pkgs = ['numpy', 'pandas', 'sklearn', 'skl2onnx', 'onnxruntime', 'onnx']
dfi = pandas.DataFrame(machine_information(pkgs))
dfi.to_csv("%s.time.csv" % filename, index=False)
print(dfi)

#########################
# Shows errors.
# +++++++++++++

err = df[~df.stderr.isnull()]
err = err[["_model", "stderr"]]
err.to_csv(filename + ".err.csv", index=False)
print(err)


#############################
# Plot the results by time
# ++++++++++++++++++++++++

df = df[df.stderr.isnull() & ~df.ratio.isnull()].sort_values("ratio").copy()
df['model'] = df['_model'].apply(lambda s: s.replace("Sklearn", ""))

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
df.plot(x="original_time", y="ratio", ax=ax,
        logx=True, logy=True, kind="scatter")
xmin, xmax = df.original_time.min(), df.original_time.max()
ax.plot([xmin, xmax], [1, 1], "--", label="1x")
ax.plot([xmin, xmax], [2, 2], "--", label="2x slower")
ax.plot([xmin, xmax], [0.5, 0.5], "--", label="2x faster")
ax.set_title("Ratio onnxruntime / scikit-learn\nLower is better")
ax.set_xlabel("execution time with scikit-learn (seconds)")
ax.set_ylabel("Ratio onnxruntime / scikit-learn\nLower is better.")
ax.legend()
fig.tight_layout()
fig.savefig("%s.xy.png" % filename)


#############################
# Plot the results by model
# +++++++++++++++++++++++++


fig, ax = plt.subplots(1, 1, figsize=(10, 25))
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
fig.savefig("%s.model.png" % filename)

# plt.show()
