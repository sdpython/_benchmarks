# coding: utf-8
"""
Benchmark of :epkg:`onnxruntime` for a cascade of scalers (additions).
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
from pymlbenchmark.benchmark import BenchPerfTest, BenchPerf
from pymlbenchmark.context import machine_information
from pymlbenchmark.datasets.artificial import random_binary_classification
from pymlbenchmark.plotting import plot_bench_results
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.onnx_ops import OnnxScaler
from onnxruntime import InferenceSession

################################
# Benchmark
# +++++++++


def generate_onnx_graph(dim, nbnode, input_name='X1'):
    """Generates a series of consecutive scalers."""

    matrices = []
    scale = list(numpy.ones((1, dim)).ravel())
    i1 = input_name
    for i in range(nbnode - 1):
        i2 = list(-random_binary_classification(1, dim)[0].ravel())
        matrices.append(i2)
        node = OnnxScaler(i1, offset=i2, scale=scale)
        i1 = node
    i2 = list(-random_binary_classification(1, dim)[0].ravel())
    matrices.append(i2)
    node = OnnxScaler(i1, offset=i2, scale=scale, output_names=['Y'])
    onx = node.to_onnx([(input_name, FloatTensorType((1, dim)))],
                       outputs=[('Y', FloatTensorType((1, dim)))])
    return onx, matrices


class GraphORtBenchPerfTest(BenchPerfTest):
    def __init__(self, dim=4, nbnode=3):
        BenchPerfTest.__init__(self)
        self.input_name = 'X1'
        self.nbnode = nbnode
        self.scale = numpy.ones((1, dim))
        self.onx, self.matrices = generate_onnx_graph(dim,
                                                      nbnode, self.input_name)
        as_string = self.onx.SerializeToString()
        try:
            self.ort = InferenceSession(as_string)
        except RuntimeError as e:
            raise RuntimeError("Issue with {}\n{}".format(
                e, self.onx))

    def fcts(self, **kwargs):

        def predict_ort(X, model=self.ort):
            return self.ort.run(None, {self.input_name: X})[0]

        def predict_npy(X, model=self.matrices, scale=self.scale):
            res = X.copy()
            for mat in model:
                res += X
                res *= scale
            return res

        return [{'lib': 'ort', 'fct': predict_ort},
                {'lib': 'npy', 'fct': predict_npy}]

    def data(self, N=10, dim=4, **kwargs):  # pylint: disable=W0221
        return tuple(o.astype(numpy.float32)
                     for o in random_binary_classification(N, dim)[:1])


@ignore_warnings(category=FutureWarning)
def run_bench(repeat=20, number=2, verbose=False):

    pbefore = dict(dim=[1, 10, 100],
                   nbnode=[1, 2, 3, 5, 10, 50, 100, 150, 200, 250, 300])
    pafter = dict(N=[1, 10, 100, 1000, 10000])

    test = lambda dim=None, **opts: GraphORtBenchPerfTest(dim=dim, **opts)
    bp = BenchPerf(pbefore, pafter, test)

    start = time()
    results = list(bp.enumerate_run_benchs(repeat=repeat, verbose=verbose,
                                           number=number))
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

pkgs = ['numpy', 'pandas', 'sklearn', 'skl2onnx', 'onnxruntime', 'onnx']
dfi = pandas.DataFrame(machine_information(pkgs))
dfi.to_csv("%s.time.csv" % filename, index=False)
print(dfi)


#############################
# Plot the results by number of nodes
# +++++++++++++++++++++++++++++++++++

from pymlbenchmark.plotting import plot_bench_results
plot_bench_results(df, row_cols='N', col_cols='dim',
                   x_value='nbnode',
                   title="%s\nBenchmark scikit-learn / onnxruntime" % "Cascade Scaler")

plt.savefig("%s.node.png" % filename)


##################################
# Plot one ONNX graph
# +++++++++++++++++++

for nbnode in (2, 4):
    onx = generate_onnx_graph(5, nbnode)[0]

    from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer
    pydot_graph = GetPydotGraph(onx.graph, name=onx.graph.name, rankdir="TB",
                                node_producer=GetOpNodeProducer("docstring"))
    pydot_graph.write_dot("graph.%d.dot" % nbnode)
    os.system('dot -O -Tpng graph.%d.dot' % nbnode)
image = plt.imread("graph.%d.dot.png" % nbnode)
plt.imshow(image)
plt.axis('off')
plt.savefig("%s.%d.node.png" % (filename, nbnode))
