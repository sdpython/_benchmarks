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
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.extmath import softmax
from scipy.special import expit
from pymlbenchmark.context import machine_information
from pymlbenchmark.benchmark import BenchPerf, ProfilerCall
from pymlbenchmark.external import OnnxRuntimeBenchPerfTestBinaryClassification
from pymlbenchmark.plotting import plot_bench_results

model_name = "MLPClassifier"
filename = os.path.splitext(os.path.split(__file__)[-1])[0]


@ignore_warnings(category=FutureWarning)
def run_bench(repeat=10, verbose=False):

    pbefore = dict(hidden_layer_sizes=[(2,), (10,), (20,),
                                       (2, 2), (10, 2), (20, 2)],
                   activation=['relu', 'logistic'],
                   dim=[2, 5, 10],
                   onnx_options=[{MLPClassifier: {'zipmap': False}}])
    pafter = dict(N=[1, 10, 100, 1000])

    merged = {}
    merged.update(pbefore)
    merged.update(pafter)
    d0 = {k: v[0] for k, v in merged.items()}

    profilers = [ProfilerCall(d0, module="cProfile"),
                 ProfilerCall(d0, module="cProfile")]

    test = lambda dim=None, **opts: OnnxRuntimeBenchPerfTestBinaryClassification(
        MLPClassifier, dim=dim, **opts)

    bp = BenchPerf(pbefore, pafter, test, profilers=profilers)

    with sklearn.config_context(assume_finite=True):
        start = time()
        results = list(bp.enumerate_run_benchs(repeat=repeat, verbose=verbose,
                                               stop_if_error=False))
        end = time()

    results_df = pandas.DataFrame(results)
    print("Total time = %0.3f sec\n" % (end - start))
    return results_df, profilers


#########################
# Runs the benchmark
# ++++++++++++++++++

df, profilers = run_bench(verbose=True)
df.to_csv("%s.perf.csv" % filename, index=False)
print(df.head())

with open("%s.prof.txt" % filename, "w") as f:
    for prof in profilers:
        f.write("\n#########################################\n\n")
        prof.to_txt(f)

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
    la = la.replace("fit_intercept", "fi")
    la = la.replace("True", "1")
    la = la.replace("False", "0")
    la = la.replace("max_depth", "mxd")
    return la


plot_bench_results(df, row_cols=['N', 'hidden_layer_sizes'],
                   col_cols='method',
                   hue_cols='activation',
                   cmp_col_values=('lib', 'skl'),
                   x_value='dim', y_value='mean',
                   title="%s\nBenchmark scikit-learn / onnxruntime" % model_name,
                   label_fct=label_fct)
plt.savefig("%s.png" % filename)

import sys
if "--quiet" not in sys.argv:
    plt.show()
