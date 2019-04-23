# coding: utf-8
"""
Benchmark of :epkg:`onnxruntime` to study time spent from beginning
to every node.
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
from collections import OrderedDict
import numpy
import pandas
import matplotlib.pyplot as plt
from sklearn.utils.testing import ignore_warnings
from sklearn.utils.extmath import softmax
from sklearn.neural_network import MLPClassifier
from pyquickhelper.loghelper import run_cmd, sys_path_append
from pymlbenchmark.benchmark import BenchPerfTest, BenchPerf
from pymlbenchmark.context import machine_information
from pymlbenchmark.datasets.artificial import random_binary_classification
from pymlbenchmark.plotting import plot_bench_results
from onnxruntime import InferenceSession
from skonnxrt.helpers.onnx_helper import enumerate_model_node_outputs
from skonnxrt.sklapi import OnnxTransformer

################################
# Trains a MLPClassifier
# ++++++++++++++++++++++
#
# Training

fixed_dim = 10
X, y = random_binary_classification(N=10000, dim=fixed_dim)
model = MLPClassifier(hidden_layer_sizes=(10, 10), activation='logistic')
model.fit(X, y)

# Converting
from skl2onnx import to_onnx
model_onnx = to_onnx(model, X.astype(numpy.float32))

##################################
# Display the ONNX graph
# ++++++++++++++++++++++

if not os.path.exists("pipeline_mlp.dot.png"):
    from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer
    pydot_graph = GetPydotGraph(model_onnx.graph, name=model_onnx.graph.name, rankdir="TB",
                                node_producer=GetOpNodeProducer("docstring", color="yellow",
                                                                fillcolor="yellow", style="filled"))
    pydot_graph.write_dot("pipeline_mlp.dot")

    import os
    os.system('dot -O -Gdpi=300 -Tpng pipeline_mlp.dot')

    import matplotlib.pyplot as plt
    image = plt.imread("pipeline_mlp.dot.png")
    fig, ax = plt.subplots(figsize=(40, 20))
    ax.imshow(image)
    ax.axis('off')

##################################
# Intermediate outputs
# ++++++++++++++++++++

output_names = list(enumerate_model_node_outputs(model_onnx))

##################################
# Bench definition
# ++++++++++++++++


class GraphORtBenchPerfTest(BenchPerfTest):

    def __init__(self, fixed_dim=None, skl_model=None, model_onnx=None):
        BenchPerfTest.__init__(self)
        if fixed_dim is None:
            raise RuntimeError("fixed_dim cannot be None.")
        if skl_model is None:
            raise RuntimeError("skl_model cannot be None.")
        if model_onnx is None:
            raise RuntimeError("model_onnx cannot be None.")
        self.fixed_dim = fixed_dim
        self.skl_model = skl_model
        self.model_onnx = model_onnx

        output_names = list(enumerate_model_node_outputs(model_onnx))
        self.onnx_bytes = model_onnx.SerializeToString()

        models = OrderedDict()
        for name in output_names:
            models[name] = OnnxTransformer(self.onnx_bytes, name)
            models[name].fit()
        self.onnx_models = models

    def fcts(self, **kwargs):

        sess = InferenceSession(self.onnx_bytes)
        fcts = [{'lib': 'skl', 'method': 'skl_proba', 'fct': self.skl_model.predict},
                {'lib': 'ort', 'method': 'onnx_proba',
                 'fct': lambda X, sess=sess: sess.run(None, {'X': X.astype(numpy.float32)})[0]}]
        for k, v in self.onnx_models.items():
            def fct(X, onx=v.onnxrt_): return onx.run(
                None, {'X': X.astype(numpy.float32)})[0]
            fcts.append(dict(lib='ort', method='ox_' + k, fct=fct))

        return fcts

    def data(self, N=10, dim=4, **kwargs):  # pylint: disable=W0221
        if self.fixed_dim != dim:
            raise RuntimeError(
                "Only dim={} is allowed not {}.".format(self.fixed_dim, dim))
        return tuple(o.astype(numpy.float32)
                     for o in random_binary_classification(N, dim)[:1])

#######################
# Run bench
# +++++++++


@ignore_warnings(category=FutureWarning)
def run_bench(repeat=25, number=20, verbose=False,
              fixed_dim=None, skl_model=None, model_onnx=None):

    pbefore = dict(dim=[fixed_dim],
                   fixed_dim=[fixed_dim],
                   skl_model=[skl_model],
                   model_onnx=[model_onnx])
    pafter = dict(N=[1, 2, 5, 10, 20, 50, 100, 200, 500,
                     1000, 2000, 5000, 10000
                     ])

    test = lambda dim=None, **opts: GraphORtBenchPerfTest(**opts)
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
df = run_bench(verbose=True, fixed_dim=fixed_dim, skl_model=model,
               model_onnx=model_onnx)
df = df.drop(['model_onnx', 'skl_model'], axis=1)
df.to_csv("%s.perf.csv" % filename, index=False)
print(df.head())

#########################
# Extracts information about the machine used
# +++++++++++++++++++++++++++++++++++++++++++

pkgs = ['numpy', 'pandas', 'sklearn', 'skl2onnx', 'onnxruntime',
        'onnx', 'skonnxrt']
dfi = pandas.DataFrame(machine_information(pkgs))
dfi.to_csv("%s.time.csv" % filename, index=False)
print(dfi)


#############################
# Plot the results by number of nodes
# +++++++++++++++++++++++++++++++++++

if False:
    from pymlbenchmark.plotting import plot_bench_results
    plot_bench_results(df, row_cols='N', col_cols='dim',
                       x_value='nbnode',
                       title="%s\nBenchmark scikit-learn / onnxruntime" % "Cascade Scaler")

    plt.savefig("%s.node.png" % filename)
