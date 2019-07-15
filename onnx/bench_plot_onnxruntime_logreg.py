# coding: utf-8
"""
Benchmark of :epkg:`onnxruntime` on LogisticRegression.
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
from sklearn.linear_model import LogisticRegression
from sklearn.utils.testing import ignore_warnings
from sklearn.utils.extmath import softmax
from scipy.special import expit
from pymlbenchmark.context import machine_information
from pymlbenchmark.benchmark import BenchPerf
from pymlbenchmark.external import OnnxRuntimeBenchPerfTestBinaryClassification
from pymlbenchmark.plotting import plot_bench_results

model_name = "LogisticRegression"
filename = os.path.splitext(os.path.split(__file__)[-1])[0]


class OnnxRuntimeBenchPerfTestBinaryClassification3(OnnxRuntimeBenchPerfTestBinaryClassification):
    """
    Overwrites the class to add a pure python implementation
    of the logistic regression.
    """

    def fcts(self, dim=None, **kwargs):

        def predict_py_predict(X, model=self.skl):
            coef = model.coef_
            intercept = model.intercept_
            pred = numpy.dot(X, coef.T) + intercept
            return (pred >= 0).astype(numpy.int32)

        def predict_py_predict_proba(X, model=self.skl):
            coef = model.coef_
            intercept = model.intercept_
            pred = numpy.dot(X, coef.T) + intercept
            decision_2d = numpy.c_[-pred, pred]
            return expit(decision_2d)

        res = OnnxRuntimeBenchPerfTestBinaryClassification.fcts(
            self, dim=dim, **kwargs)
        res.extend([
            {'method': 'predict', 'lib': 'py', 'fct': predict_py_predict},
            {'method': 'predict_proba', 'lib': 'py',
                'fct': predict_py_predict_proba},
        ])
        return res


@ignore_warnings(category=FutureWarning)
def run_bench(repeat=100, verbose=False):

    pbefore = dict(dim=[1, 5, 10, 20, 50, 100, 150],
                   fit_intercept=[True])
    pafter = dict(N=[1, 10, 100, 1000, 10000])
    test = lambda dim=None, **opts: OnnxRuntimeBenchPerfTestBinaryClassification3(
        LogisticRegression, dim=dim, **opts)
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
    la = la.replace("onxpython", "opy")
    la = la.replace("onxonnxruntime1", "ort")
    la = la.replace("fit_intercept", "fi")
    la = la.replace("True", "1")
    la = la.replace("False", "0")
    la = la.replace("max_depth", "mxd")
    return la


plot_bench_results(df, row_cols='N', col_cols='method',
                   x_value='dim', hue_cols='fit_intercept',
                   title="%s\nBenchmark scikit-learn / onnxruntime" % model_name,
                   label_fct=label_fct)
plt.savefig("%s.png" % filename)
# plt.show()
