# coding: utf-8
"""
Benchmark of onnxruntime on MultinomialNB.
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
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.testing import ignore_warnings
from sklearn.utils.extmath import softmax
from scipy.special import expit
from pymlbenchmark.context import machine_information
from pymlbenchmark.benchmark import BenchPerf
from pymlbenchmark.external import OnnxRuntimeBenchPerfTestBinaryClassification
from pymlbenchmark.plotting import plot_bench_results

model_name = "MultinomialNB"
filename = os.path.splitext(os.path.split(__file__)[-1])[0]


@ignore_warnings(category=FutureWarning)
def run_bench(repeat=100, verbose=False):

    pbefore = dict(dim=[1, 5, 10, 20, 50, 100, 200],
                   alpha=[0., 0.5, 1.],
                   fit_prior=[True, False])
    pafter = dict(N=[1, 10])

    test = lambda dim=None, **opts: OnnxRuntimeBenchPerfTestBinaryClassification(
        MultinomialNB, dim=dim, **opts)
    bp = BenchPerf(pbefore, pafter, test)

    start = time()
    results = list(bp.enumerate_run_benchs(repeat=repeat, verbose=verbose))
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

pkgs = ['numpy', 'pandas', 'sklearn', 'skl2onnx', 'onnxruntime', 'onnx']
dfi = pandas.DataFrame(machine_information(pkgs))
dfi.to_csv("%s.time.csv" % filename, index=False)
print(dfi)

#############################
# Plot the results
# ++++++++++++++++

plot_bench_results(df, row_cols=['N', 'fit_priors'], col_cols='method',
                   x_value='dim', hue_cols=['alphas'],
                   title="%s\nBenchmark scikit-learn / onnxruntime" % model_name)
plt.savefig("%s.png" % filename)
# plt.show()