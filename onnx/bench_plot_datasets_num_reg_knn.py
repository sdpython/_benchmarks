# coding: utf-8
"""
Benchmark of :epkg:`onnxruntime` on a couple of datasets and
models.
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
from sklearn.datasets import load_boston, load_diabetes, make_regression
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,
    HistGradientBoostingRegressor
)
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR, NuSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.testing import ignore_warnings
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from mlprodict.onnxrt import OnnxInference
from pymlbenchmark.context import machine_information
from pymlbenchmark.benchmark import BenchPerf, BenchPerfTest
from pymlbenchmark.plotting import plot_bench_results, plot_bench_xtime
from skl2onnx import to_onnx
from onnxruntime import InferenceSession
from mlprodict.onnx_conv import register_converters, register_rewritten_operators
from mlprodict.tools.model_info import analyze_model
from mlprodict.tools.asv_options_helper import get_opset_number_from_onnx

register_converters()
register_rewritten_operators()

filename = os.path.splitext(os.path.split(__file__)[-1])[0]


def create_datasets():
    results = {}
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    results['boston'] = [X_train, X_test, y_train, y_test]
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    results['diabetes'] = [X_train, X_test, y_train, y_test]

    X, y = make_regression(1000, 10)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    results['rndbin100'] = [X_train, X_test, y_train, y_test]

    X, y = make_regression(1000, 10, n_targets=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    results['rnd3cl100'] = [X_train, X_test, y_train, y_test]
    return results


def get_model(model_name):
    if model_name in ("LR", "LR-ZM"):
        return LinearRegression()
    if model_name == "DT":
        return DecisionTreeRegressor(max_depth=6)
    if model_name == 'HGB':        
        return HistGradientBoostingRegressor(max_depth=6, max_iter=100)
    if model_name == "RF":
        return RandomForestRegressor(max_depth=6, n_estimators=100)
    if model_name == "GBT":
        return GradientBoostingRegressor(max_depth=6, n_estimators=100)
    if model_name in ("KNN", "KNN-cdist"):
        return KNeighborsRegressor(algorithm='brute')
    if model_name == "MLP":
        return MLPRegressor()
    if model_name == "ADA":
        return AdaBoostRegressor()
    if model_name == "SVR":
        return SVR()
    if model_name == "NuSVR":
        return NuSVR()
    if model_name == "XGB":
        return XGBRegressor(max_depth=6, n_estimators=100)
    if model_name == "LGB":
        return LGBMRegressor(max_depth=6, n_estimators=100)
    raise ValueError("Unknown model name '{}'.".format(model_name))


common_datasets = create_datasets()


class DatasetsOrtBenchPerfTest(BenchPerfTest):
    def __init__(self, model, dataset, norm):
        BenchPerfTest.__init__(self)
        self.model_name = model
        self.dataset_name = dataset
        self.datas = common_datasets[dataset]
        skl_model = get_model(model)
        if norm:
            if 'NB' in model:
                self.model = make_pipeline(MinMaxScaler(), skl_model)
            else:
                self.model = make_pipeline(StandardScaler(), skl_model)
        else:
            self.model = skl_model
        self.model.fit(self.datas[0], self.datas[2])
        self.data_test = self.datas[1]

        if '-cdist' in model:
            options = {id(skl_model): {'optim': 'cdist'}}
        elif "-ZM" in model:
            options = {id(skl_model): {'zipmap': False}}
        else:
            options = None
        self.onx = to_onnx(self.model, self.datas[0].astype(
            numpy.float32), options=options,
            target_opset=get_opset_number_from_onnx())
        logger = getLogger("skl2onnx")
        logger.propagate = False
        logger.disabled = True
        self.ort = InferenceSession(self.onx.SerializeToString())
        self.oinf = OnnxInference(self.onx, runtime='python')
        self.oinfc = OnnxInference(self.onx, runtime='python_compiled')
        self.output_name = self.oinf.output_names[-1]
        self.input_name = self.ort.get_inputs()[0].name
        self.model_info = analyze_model(self.model)

    def fcts(self, **kwargs):

        def predict_ort(X, model=self.ort, namei=self.input_name):
            try:
                return model.run(None, {namei: X})[1]
            except Exception as e:
                return None

        def predict_skl(X, model=self.model):
            return model.predict(X)

        def predict_pyrt(X, model=self.oinf, namei=self.input_name,
                         nameo=self.output_name):
            return model.run({namei: X})[nameo]

        def predict_pyrtc(X, model=self.oinfc, namei=self.input_name,
                          nameo=self.output_name):
            return model.run({namei: X})[nameo]

        return [{'lib': 'ort', 'fct': predict_ort},
                {'lib': 'skl', 'fct': predict_skl},
                {'lib': 'pyrt', 'fct': predict_pyrt},
                {'lib': 'pyrtc', 'fct': predict_pyrtc}]

    def data(self, N=10, dim=-1, **kwargs):  # pylint: disable=W0221
        if dim != -1:
            raise ValueError("dim must be -1 as it is fixed.")

        nbs = numpy.random.randint(0, self.data_test.shape[0] - 1, N)
        res = self.data_test[nbs, :].astype(numpy.float32)
        return (res, )

    def validate(self, results, **kwargs):
        nb = 5
        if len(results) != nb * 4:  # skl, ort, pyrt, pyrtc
            raise RuntimeError(
                "Expected only 3 results not {0}.".format(len(results)))
        res = {}
        for idt, fct, vals in results:
            if idt not in res:
                res[idt] = {}
            if isinstance(vals, list):
                vals = pandas.DataFrame(vals).values
            lib = fct['lib']
            res[idt][lib] = vals

        if len(res) != nb:
            raise RuntimeError(
                "Expected only 2 results not {0}.".format(len(results)))
        final = {}
        for diff_name in ['ort', 'pyrt', 'pyrtc']:
            diffs = []
            for i in range(0, nb):
                r = res[i]
                if diff_name not in r or r[diff_name] is None:
                    continue
                bas = numpy.squeeze(r['skl'])
                onn = numpy.squeeze(r[diff_name].squeeze())
                if bas.shape != onn.shape:
                    raise AssertionError("Shape mismatch {} != {} params={}".format(
                        bas.shape, onn.shape, results[0][0]))
                diff = numpy.max(numpy.abs(onn - bas))
                diffs.append(diff)
            if len(diffs) > 0:
                final.update({'diff_%s' % diff_name: sum(diffs) / nb,
                              'upper_diff_%s' % diff_name: max(diffs),
                              'lower_diff_%s' % diff_name: min(diffs)})
        for k, v in self.model_info.items():
            final['fit_' + k] = v
        return final


@ignore_warnings(category=FutureWarning)
def run_bench(repeat=5, verbose=False):

    pbefore = dict(dim=[-1],
                   model=list(sorted(['XGB', 'LGB',
                                      'SVR', 'NuSVR',
                                      'RF', 'DT',
                                      'ADA', 'MLP', 'LR-ZM',
                                      'LR', 'GBT', 'KNN',
                                      'KNN-cdist', 'HGB'])),
                   norm=[False, True],
                   dataset=["boston", "diabetes", "rndbin100"])
    pafter = dict(N=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000,
                     2000, 5000, 10000, 20000, 50000])

    test = lambda dim=None, **opts: DatasetsOrtBenchPerfTest(**opts)
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
    la = la.replace("onxpython_compiled", "opyc")
    la = la.replace("onxpython", "opy")
    la = la.replace("onxonnxruntime1", "ort")
    la = la.replace("fit_intercept", "fi")
    la = la.replace("True", "1")
    la = la.replace("False", "0")
    la = la.replace("max_depth", "mxd")
    return la


plot_bench_xtime(df[df.norm], col_cols='dataset',
                 hue_cols='model',
                 title="Numerical datasets - norm=False\nBenchmark scikit-learn / onnxruntime")
plt.savefig("%s.normT.time.png" % filename)
# plt.show()

plot_bench_xtime(df[~df.norm], col_cols='dataset',
                 hue_cols='model',
                 title="Numerical datasets - norm=False\nBenchmark scikit-learn / onnxruntime")
plt.savefig("%s.normF.time.png" % filename)
# plt.show()

plot_bench_results(df, row_cols='model', col_cols=('dataset', 'norm'),
                   x_value='N',
                   title="Numerical datasets\nBenchmark scikit-learn / onnxruntime")
plt.savefig("%s.curve.png" % filename)
# plt.show()
