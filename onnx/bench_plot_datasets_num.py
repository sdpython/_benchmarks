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
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.utils.testing import ignore_warnings
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from pymlbenchmark.context import machine_information
from pymlbenchmark.benchmark import BenchPerf, BenchPerfTest
from pymlbenchmark.plotting import plot_bench_results, plot_bench_xtime
from skl2onnx import to_onnx
from onnxruntime import InferenceSession


filename = os.path.splitext(os.path.split(__file__)[-1])[0]


def create_datasets():
    from sklearn.datasets import load_breast_cancer, load_digits
    results = {}
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    results['breast_cancer'] = [X_train, X_test, y_train, y_test]
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    results['digits'] = [X_train, X_test, y_train, y_test]
    return results


def get_model(model_name):
    if model_name == "LR":
        return LogisticRegression(solver="liblinear", penalty="l2")
    elif model_name == "DT":
        return DecisionTreeClassifier(max_depth=4)
    elif model_name == "RF":
        return RandomForestClassifier(max_depth=4, n_estimators=10)
    elif model_name == "GBT":
        return GradientBoostingClassifier(max_depth=4, n_estimators=10)
    elif model_name == "KNN":
        return KNeighborsClassifier()
    elif model_name == "MLP":
        return MLPClassifier()
    elif model_name == "MNB":
        return MultinomialNB()
    elif model_name == "BNB":
        return BernoulliNB()
    elif model_name == "ADA":
        return AdaBoostClassifier()
    elif model_name == "SVC":
        return SVC(probability=True)
    elif model_name == "NuSVC":
        return NuSVC(probability=True)
    else:
        raise ValueError("Unknown model name '{}'.".format(model_name))


common_datasets = create_datasets()


class DatasetsOrtBenchPerfTest(BenchPerfTest):
    def __init__(self, model, dataset, norm):
        BenchPerfTest.__init__(self)
        self.model_name = model
        self.dataset_name = dataset
        self.datas = common_datasets[dataset]
        if norm:
            self.model = make_pipeline(StandardScaler(), get_model(model))
        else:
            self.model = get_model(model)
        self.model.fit(self.datas[0], self.datas[2])
        self.data_test = self.datas[1]

        self.onx = to_onnx(self.model, self.datas[0].astype(numpy.float32))
        logger = getLogger("skl2onnx")
        logger.propagate = False
        logger.disabled = True
        self.ort = InferenceSession(self.onx.SerializeToString())
        self.broadcast = self.model_name not in {'KNN'}

    def fcts(self, **kwargs):

        def predict_ort(X, model=self.ort):
            input_name = self.ort.get_inputs()[0].name
            if self.broadcast:
                return model.run(None, {input_name: X})[1]
            else:
                return [model.run(None, {input_name: X[i]})[1][0] for i in range(0, X.shape[0])]

        def predict_skl(X, model=self.model):
            return model.predict_proba(X)

        return [{'lib': 'ort', 'fct': predict_ort},
                {'lib': 'skl', 'fct': predict_skl}]

    def data(self, N=10, dim=-1, **kwargs):  # pylint: disable=W0221
        if dim != -1:
            raise ValueError("dim must be -1 as it is fixed.")

        nbs = numpy.random.randint(0, self.data_test.shape[0] - 1, N)
        res = self.data_test[nbs, :].astype(numpy.float32)
        return (res, )

    def validate(self, results, **kwargs):
        nb = 5
        if len(results) != nb * 2:
            raise RuntimeError(
                "Expected only 2 results not {0}.".format(len(results)))
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
        diffs = []
        for i in range(0, nb):
            r = res[i]
            bas = r['skl']
            onn = r['ort']
            if bas.shape != onn.shape:
                raise AssertionError("Shape mismatch {} != {} params={}".format(
                    bas.shape, onn.shape, results[0][0]))
            diff = numpy.max(numpy.abs(onn - bas))
            diffs.append(diff)
        return {'diff': sum(diffs) / nb,
                'upper_diff': max(diffs),
                'lower_diff': min(diffs)}


@ignore_warnings(category=FutureWarning)
def run_bench(repeat=5, verbose=False):

    pbefore = dict(dim=[-1],
                   model=list(sorted(['SVC', 'NuSVC', 'BNB',
                                      'RF', 'DT', 'MNB',
                                      'ADA', 'MLP',
                                      'LR', 'GBT', 'KNN'])),
                   norm=[False, True],
                   dataset=["breast_cancer", "digits"])
    pafter = dict(N=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000,
                     2000, 5000, 10000, 20000, 50000])

    test = lambda dim=None, **opts: DatasetsOrtBenchPerfTest(**opts)
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

plot_bench_results(df, row_cols='model', col_cols=('dataset', 'norm'),
                   x_value='N', y_value='diff',
                   err_value=('lower_diff', 'upper_diff'),
                   title="Numerical datasets\Absolute difference scikit-learn / onnxruntime")
plt.savefig("%s.diff.png" % filename)
# plt.show()
