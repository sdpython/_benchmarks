# coding: utf-8
"""
Benchmark of :epkg:`onnxruntime` for
`ReduceSum <https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceSum>`_.
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
import sklearn
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.extmath import softmax
from pyquickhelper.loghelper import run_cmd, sys_path_append
from pymlbenchmark.benchmark import BenchPerfTest, BenchPerf
from pymlbenchmark.context import machine_information
from pymlbenchmark.plotting import plot_bench_results
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.onnx_ops import OnnxReduceSum
from onnxruntime import InferenceSession
from onnxruntime.capi.onnxruntime_pybind11_state import Fail
from mlprodict.onnxrt import OnnxInference
from mlprodict.tools.asv_options_helper import (
    get_opset_number_from_onnx, get_ir_version_from_onnx)

################################
# Benchmark
# +++++++++


def generate_onnx_graph(edims, axes, input_name='X'):
    """Generates a series of consecutive additions."""

    node = OnnxReduceSum(input_name, axes=list(axes),
                         op_version=get_opset_number_from_onnx(),
                         output_names=['Y'])
    onx = node.to_onnx([(input_name, FloatTensorType((None,) + tuple(edims)))],
                       outputs=[('Y', FloatTensorType())])
    return onx


class GraphOrtBenchPerfTest(BenchPerfTest):
    def __init__(self, edims=(1000, 1000), axes=(1, )):
        BenchPerfTest.__init__(self)
        self.input_name = 'X'
        self.edims = edims
        self.axes = axes
        self.onx = generate_onnx_graph(edims, axes, self.input_name)
        as_string = self.onx.SerializeToString()
        try:
            self.ort = InferenceSession(as_string)
        except Fail as e:
            raise RuntimeError(
                "Issue\n{}".format(self.onx)) from e
        self.rtpy = OnnxInference(as_string, runtime='python_compiled')

    def fcts(self, **kwargs):

        def predict_ort(X, model=self.ort):
            return self.ort.run(None, {self.input_name: X})[0]

        def predict_rtpy(X, model=self.ort):
            return self.rtpy.run({self.input_name: X})['Y']

        def predict_npy(X):
            return numpy.sum(X, axis=self.axes)

        return [{'lib': 'ort', 'fct': predict_ort},
                {'lib': 'npy', 'fct': predict_npy},
                {'lib': 'rtpy', 'fct': predict_rtpy}]

    def data(self, N=10, edims=None, **kwargs):  # pylint: disable=W0221
        new_dims = list((N,) + tuple(edims))
        return (numpy.random.rand(*new_dims).astype(numpy.float32), )


def fct_filter_test(N=None, edims=None, axes=None):
    if axes is None:
        return True
    for a in axes:
        if a > len(edims):
            print('-', N, edims, axes)
            return False    
    return True


@ignore_warnings(category=FutureWarning)
def run_bench(repeat=20, number=10, verbose=False):

    pbefore = dict(edims=[(10, 10), (100, 100), (50, 100, 10)],
                   axes=[(1, ), (2, ), (1, 2)])
    pafter = dict(N=[1, 10, 100, 1000, 2000, 5000])

    test = lambda edims=None, axes=None, **opts: GraphOrtBenchPerfTest(edims=edims, axes=axes, **opts)
    bp = BenchPerf(pbefore, pafter, test,
                   filter_test=fct_filter_test)

    with sklearn.config_context(assume_finite=True):
        start = time()
        results = list(bp.enumerate_run_benchs(repeat=repeat, verbose=verbose,
                                               number=number, stop_if_error=False))
        end = time()

    results_df = pandas.DataFrame(results)
    print("Total time = %0.3f sec\n" % (end - start))
    return results_df

#########################
# Runs the benchmark
# ++++++++++++++++++


filename = os.path.splitext(os.path.split(__file__)[-1])[0]
df = run_bench(verbose=True)
df.to_csv("%s.perf.csv" % filename, index=False)
print(df.head())

#########################
# Extracts information about the machine used
# +++++++++++++++++++++++++++++++++++++++++++

pkgs = ['numpy', 'pandas', 'sklearn', 'skl2onnx',
        'onnxruntime', 'onnx', 'mlprodict']
dfi = pandas.DataFrame(machine_information(pkgs))
dfi.to_csv("%s.time.csv" % filename, index=False)
print(dfi)


#############################
# Plot the results by number of nodes
# +++++++++++++++++++++++++++++++++++

def label_fct(la):
    la = la.replace("onxpython_compiled", "opy")
    la = la.replace("onxpython", "opy")
    la = la.replace("onxonnxruntime1", "ort")
    la = la.replace("edims=(", "edims=(N, ")
    return la


from pymlbenchmark.plotting import plot_bench_results
plot_bench_results(df, row_cols='edims', col_cols='axes',
                   x_value='N', cmp_col_values=('lib', 'npy'),
                   title="Benchmark ReduceSum",
                   label_fct=label_fct)

plt.savefig("%s.node.png" % filename)
