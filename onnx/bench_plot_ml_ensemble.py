# coding: utf-8
"""
Benchmark of a couple of machine learning frameworks
for random forest.
"""
# Authors: Xavier Dupré (benchmark)
# License: MIT
import matplotlib
matplotlib.use('Agg')

import os
from logging import getLogger
from time import perf_counter as time
import numpy
import pandas
import matplotlib.pyplot as plt
import sklearn
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils.testing import ignore_warnings
from sklearn.datasets import load_breast_cancer, load_digits
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from mlprodict.onnxrt import OnnxInference
from pymlbenchmark.context import machine_information
from pymlbenchmark.benchmark import BenchPerf, BenchPerfTest
from pymlbenchmark.plotting import plot_bench_results, plot_bench_xtime
from skl2onnx import to_onnx
from onnxruntime import InferenceSession
from mlprodict.onnx_conv import register_converters
from mlprodict.tools.model_info import analyze_model


filename = os.path.splitext(os.path.split(__file__)[-1])[0]


def create_datasets():
    results = {}

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    results['breast_cancer'] = [X_train, X_test, y_train, y_test]

    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=41)
    results['digits'] = [X_train, X_test, y_train, y_test]

    X, y = make_classification(100000)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    results['rndbin100'] = [X_train, X_test, y_train, y_test]
    return results


register_converters()
common_datasets = create_datasets()


def get_model(lib):
    if lib == "sklh":
        return HistGradientBoostingRegressor(max_depth=6, max_iter=100)
    if lib == "skl":
        return RandomForestRegressor(max_depth=6, n_estimators=100)
    if lib == 'xgb':
        return XGBRegressor(max_depth=6, n_estimators=100)
    if lib == 'lgb':
        return LGBMRegressor(max_depth=6, n_estimators=100)
    raise ValueError("Unknown library '{}'.".format(lib))


class LibOrtBenchPerfTest(BenchPerfTest):
    def __init__(self, lib, dataset):
        BenchPerfTest.__init__(self)

        logger = getLogger("skl2onnx")
        logger.propagate = False
        logger.disabled = True

        self.dataset_name = dataset
        self.lib_name = lib
        self.models = {}
        self.datas = {}
        self.onxs = {}
        self.orts = {}
        self.oinfcs = {}
        self.model_info = {}
        self.output_name = {}
        self.input_name = {}
        self.models[lib] = get_model(lib)
        self.datas[lib] = common_datasets[dataset]
        x = self.datas[lib][0]
        y = self.datas[lib][2]
        self.models[lib].fit(x, y)
        self.model_info[lib] = analyze_model(self.models[lib])
        try:
            self.onxs[lib] = to_onnx(
                self.models[lib],
                self.datas[lib][0][:1].astype(numpy.float32),
                options=None)
        except RuntimeError:
            pass

        if lib in self.onxs:
            self.orts[lib] = InferenceSession(
                self.onxs[lib].SerializeToString())
            self.oinfcs[lib] = OnnxInference(
                self.onxs[lib], runtime='python_compiled')
            self.output_name[lib] = self.oinfcs[lib].output_names[-1]
            self.input_name[lib] = self.orts[lib].get_inputs()[0].name

    def fcts(self, **kwargs):

        def predict_ort(X, model, namei):
            try:
                return model.run(None, {namei: X})[1]
            except Exception as e:
                return None

        def predict_model(X, model):
            return model.predict(X)

        def predict_pyrtc(X, model, namei, nameo):
            return model.run({namei: X})[nameo]

        res = []
        for lib in [self.lib_name]:
            res.append({'lib': lib, 'rt': '',
                        'fct': lambda X: predict_model(X, self.models[lib])})
            if lib in self.oinfcs:
                res.append({'lib': lib, 'rt': 'pyrt',
                            'fct': lambda X: predict_pyrtc(
                                X, self.oinfcs[lib], self.input_name[lib],
                                self.output_name[lib])})
            if lib in self.orts:
                res.append({'lib': lib, 'rt': 'ort',
                            'fct': lambda X: predict_ort(
                                X, self.orts[lib], self.input_name[lib])})
        return res

    def data(self, N=10, dim=-1, **kwargs):  # pylint: disable=W0221
        if dim != -1:
            raise ValueError("dim must be -1 as it is fixed.")

        lib = self.lib_name
        x = self.datas[lib][1]
        nbs = numpy.random.randint(0, x.shape[0] - 1, N)
        res = x[nbs, :].astype(numpy.float32)
        return (res, )

    def validate(self, results, **kwargs):
        final = {}
        for k, v in self.model_info[self.lib_name].items():
            final['fit_' + k] = v
        return final


@ignore_warnings(category=FutureWarning)
def run_bench(repeat=10, verbose=False):

    pbefore = dict(dim=[-1],
                   lib=['sklh', 'skl', 'xgb', 'lgb'],
                   dataset=["breast_cancer", "digits", "rndbin100"])
    pafter = dict(N=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000,
                     2000, 5000, 10000, 20000, 50000])

    test = lambda dim=None, **opts: LibOrtBenchPerfTest(**opts)
    bp = BenchPerf(pbefore, pafter, test)

    with sklearn.config_context(assume_finite=True):
        start = time()
        results = list(bp.enumerate_run_benchs(repeat=repeat, verbose=verbose,
                                               stop_if_error=False))
        end = time()

    results_df = pandas.DataFrame(results)
    print("Total time = %0.3f sec\n" % (end - start))
    return results_df


if True:
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
else:
    df = pandas.read_csv("%s.perf.csv" % filename)
    print(df.head())

#############################
# Plot the results
# ++++++++++++++++


def label_fct(la):
    la = la.replace("-lib=", "")
    la = la.replace("rt=", "-")
    return la


plot_bench_results(df, row_cols=('rt',), col_cols=('dataset', ), label_fct=label_fct,
                   x_value='N', hue_cols=('lib',), cmp_col_values='lib',
                   title="Numerical datasets\nBenchmark scikit-learn, xgboost, lightgbm")
plt.savefig("%s.curve.png" % filename)
plt.show()
