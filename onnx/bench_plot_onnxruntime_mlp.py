# coding: utf-8
"""
Benchmark of :epkg:`onnxruntime` on MLPClassifier.
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
from sklearn.neural_network import MLPClassifier
from sklearn.utils.testing import ignore_warnings
from sklearn.utils.extmath import softmax
from scipy.special import expit
from pymlbenchmark.context import machine_information
from pymlbenchmark.benchmark import BenchPerf
from pymlbenchmark.external import OnnxRuntimeBenchPerfTestBinaryClassification
from pymlbenchmark.plotting import plot_bench_results

model_name = "MLPClassifier"
filename = os.path.splitext(os.path.split(__file__)[-1])[0]


@ignore_warnings(category=FutureWarning)
def run_bench(repeat=10, verbose=False):

    pbefore = dict(hidden_layer_sizes=[(2,), (10,), (20,),
                                       (2, 2), (10, 2), (20, 2)],
                   activation=['relu', 'logistic'],
                   dim=[2, 5, 10])
    pafter = dict(N=[1, 10, 100, 1000])

    test = lambda dim=None, **opts: OnnxRuntimeBenchPerfTestBinaryClassification(
        MLPClassifier, dim=dim, **opts)
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

plot_bench_results(df, row_cols=['N', 'hidden_layer_sizes'],
                   col_cols='method',
                   hue_cols='activation',
                   cmp_col_values=('lib', 'skl'),
                   x_value='dim', y_value='mean',
                   title="%s\nBenchmark scikit-learn / onnxruntime" % model_name)
plt.savefig("%s.png" % filename)
# plt.show()
