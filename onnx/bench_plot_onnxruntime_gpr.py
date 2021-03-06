# coding: utf-8
"""
Benchmark of :epkg:`onnxruntime` on DecisionTree.
"""
# Authors: Xavier Dupré (benchmark)
# License: MIT
import matplotlib
matplotlib.use('Agg')

import os
from time import perf_counter as time
import numpy
import pandas
import matplotlib.pyplot as plt
import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.extmath import softmax
from scipy.special import expit
from pymlbenchmark.context import machine_information
from pymlbenchmark.benchmark import BenchPerf
from pymlbenchmark.external import OnnxRuntimeBenchPerfTestRegression
from pymlbenchmark.plotting import plot_bench_results

model_name = "GaussianProcessRegressor"
filename = os.path.splitext(os.path.split(__file__)[-1])[0]


@ignore_warnings(category=FutureWarning)
def run_bench(repeat=10, verbose=False):

    pbefore = dict(dim=[1, 5, 10, 20], alpha=[0.1, 1., 10.],
                   onnx_options=[
                       None, {GaussianProcessRegressor: {'optim': 'cdist'}}],
                   dtype=[numpy.float32, numpy.float64])
    pafter = dict(N=[1, 10, 100, 1000])

    test = lambda dim=None, **opts: OnnxRuntimeBenchPerfTestRegression(
        GaussianProcessRegressor, dim=dim, N_fit=100, **opts)
    bp = BenchPerf(pbefore, pafter, test)

    with sklearn.config_context(assume_finite=True):
        start = time()
        results = list(bp.enumerate_run_benchs(repeat=repeat, verbose=verbose,
                                               stop_if_error=False))
        end = time()

    results_df = pandas.DataFrame(results)
    print("Total time = %0.3f sec\n" % (end - start))
    return results_df


#########################
# Runs the benchmark
# ++++++++++++++++++

df = run_bench(verbose=True)
df.to_csv("%s.perf.csv" % filename, index=False)
print(df.head())

#########################
# Extract information about the machine used
# ++++++++++++++++++++++++++++++++++++++++++

pkgs = ['numpy', 'pandas', 'sklearn', 'skl2onnx',
        'onnxruntime', 'onnx', 'mlprodict']
dfi = pandas.DataFrame(machine_information(pkgs))
dfi.to_csv("%s.time.csv" % filename, index=False)
print(dfi)

#############################
# Plot the results
# ++++++++++++++++


def label_fct(la):
    la = la.replace("onxpython_compiled", "opy")
    la = la.replace("onxpython", "opy")
    la = la.replace("onxonnxruntime1", "ort")
    la = la.replace("True", "1")
    la = la.replace("False", "0")
    la = la.replace("max_depth", "mxd")
    la = la.replace("method=predict", "cl")
    la = la.replace("method=proba", "prob")
    la = la.replace("onnx_options={<class 'sklearn.gaussian_process._gpr.GaussianProcessRegressor'>: {'optim': 'cdist'}}",
                    'cdist')
    la = la.replace("onnx_options=nan", '-')
    return la


name = "bench_plot_onnxruntime_gpr.perf.csv"
df = pandas.read_csv(name)

plot_bench_results(df, row_cols=('alpha', 'N'), col_cols='onnx_options',
                   hue_cols='dtype',
                   cmp_col_values=('lib', 'skl'),
                   x_value='dim', y_value='mean',
                   title=None, label_fct=label_fct,
                   ax=None, box_side=4)
plt.suptitle(
    "Acceleration onnxruntime / scikit-learn for GaussianProcessRegressor")

plt.savefig("%s.png" % filename)

import sys
if "--quiet" not in sys.argv:
    plt.show()
