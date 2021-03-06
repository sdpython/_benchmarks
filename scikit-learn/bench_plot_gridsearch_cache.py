# coding: utf-8
"""
Benchmark of grid search using caching.
"""
# Authors: Xavier Dupré (benchmark)
# License: MIT
from time import time
from itertools import combinations, chain
from itertools import combinations_with_replacement as combinations_w_r

import numpy as np
from numpy.random import rand
from numpy.testing import assert_almost_equal
import matplotlib.pyplot as plt
import pandas
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.utils._testing import ignore_warnings
import sklearn.utils
from mlinsights.mlbatch import PipelineCache, MLCache

##############################
# Starts a dask cluster
# +++++++++++++++++++++

has_dask = False

"""
# Does work yet.
from distributed import Client, LocalCluster
try:
    cluster = LocalCluster()
    print(cluster)
    client = Client(cluster)
    print(client)
    has_dask = True
except Exception as e:
    print("Cannot use dask due to {0}.".format(e))
    has_dask = False
"""

##############################
# Implementations to benchmark
# ++++++++++++++++++++++++++++

from pymlbenchmark.benchmark import BenchPerf, BenchPerfTest
from pymlbenchmark.datasets import random_binary_classification


class GridSearchBenchPerfTest(BenchPerfTest):

    def __init__(self, dim=None, n_jobs=1, **opts):
        assert dim is not None
        BenchPerfTest.__init__(self, **opts)
        self.n_jobs = n_jobs

    def _make_model(self, dim, cache, n_jobs):
        if cache in (None, 'no'):
            cl = Pipeline
            ps = dict()
        elif cache == 'joblib':
            cl = Pipeline
            ps = dict(memory='jb-%d-%d' % (dim, n_jobs))
        elif cache == 'dask':
            cl = Pipeline
            ps = dict(memory='dk-%d-%d' % (dim, n_jobs))
        elif cache == "memory":
            cl = PipelineCache
            ps = dict(cache_name='memory-%d-%d' % (dim, n_jobs))
        else:
            raise ValueError("Unknown cache value: '{0}'.".format(cache))

        model = cl([
            ('scale', MinMaxScaler()),
            ('pca', PCA(2)),
            ('poly', PolynomialFeatures()),
            ('bins', KBinsDiscretizer()),
            ('lr', LogisticRegression(solver='liblinear'))],
            **ps)
        params_grid = {
            'scale__feature_range': [(0, 1)],
            'pca__n_components': [2, 4],
            'poly__degree': [2, 3],
            'bins__n_bins': [5],
            'bins__encode': ["onehot-dense", "ordinal"],
            'lr__penalty': ['l1', 'l2'],
        }

        return GridSearchCV(model, params_grid, n_jobs=n_jobs, verbose=0)

    def data(self, N=None, dim=None, **opts):
        # The benchmark requires a new datasets each time.
        assert N is not None
        assert dim is not None
        return random_binary_classification(N, dim)

    def fcts(self, dim=None, **kwargs):
        # The function returns the prediction functions to tests.
        global has_dask
        options = ['no', 'joblib', 'memory']
        if has_dask:
            options.append('dask')
        models = {}
        for cache in options:
            models[cache] = self._make_model(dim, cache, self.n_jobs)

        def fit_model(X, y, cache):
            if cache == "joblib":
                sklearn.utils.parallel_backend("loky", self.n_jobs)
            elif cache == "dask":
                sklearn.utils.parallel_backend("dask", self.n_jobs)
            else:
                sklearn.utils.parallel_backend("threading", self.n_jobs)
            model = models[cache]
            model.fit(X, y)
            if cache == 'memory':
                MLCache.remove_cache(model.best_estimator_.cache_name)

        res = []
        for cache in sorted(models):
            res.append({'test': cache, 'fct': lambda X,
                        y, c=cache: fit_model(X, y, c)})
        return res


##############################
# Benchmark
# +++++++++


@ignore_warnings(category=(FutureWarning, UserWarning, DeprecationWarning))
def run_bench(repeat=3, verbose=False, number=1):
    pbefore = dict(dim=[10, 15])
    pafter = dict(N=[100, 1000, 10000], n_jobs=[1, 3])

    bp = BenchPerf(pbefore, pafter, GridSearchBenchPerfTest)

    with sklearn.config_context(assume_finite=True):
        start = time()
        results = list(bp.enumerate_run_benchs(repeat=repeat, verbose=verbose))
        end = time()

    results_df = pandas.DataFrame(results)
    print("Total time = %0.3f sec\n" % (end - start))
    return results_df


##############################
# Run the benchmark
# +++++++++++++++++

filename = "bench_plot_gridsearch_cache"
df = run_bench(verbose=True)
df.to_csv("%s.csv" % filename, index=False)
print(df.head())

if has_dask:
    cluster.close()

#########################
# Extract information about the machine used
# ++++++++++++++++++++++++++++++++++++++++++

from pymlbenchmark.context import machine_information
pkgs = ['numpy', 'pandas', 'sklearn']
dfi = pandas.DataFrame(machine_information(pkgs))
dfi.to_csv("%s.time.csv" % filename, index=False)
print(dfi)


#############################
# Plot the results
# ++++++++++++++++

from pymlbenchmark.plotting import plot_bench_results
print(df.columns)
plot_bench_results(df, row_cols=['N'],
                   col_cols=['n_jobs'], x_value='dim',
                   hue_cols=['test'],
                   cmp_col_values='test',
                   title="GridSearchCV\nBenchmark caching strategies")
import sys
if "--quiet" not in sys.argv:
    plt.show()
