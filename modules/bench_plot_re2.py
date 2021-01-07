# coding: utf-8
"""
Benchmark of `re2 <https://github.com/facebook/pyre2>`_.
"""
# Authors: Xavier Dupré (benchmark)
# License: MIT


##############################
# Implementations to benchmark
# ++++++++++++++++++++++++++++

import re
from wrapclib import re2
import numpy
from pymlbenchmark.benchmark import BenchPerf, BenchPerfTest
from pymlbenchmark.datasets import random_binary_classification


class Re2Bench(BenchPerfTest):

    def __init__(self, dim=None, **opts):
        BenchPerfTest.__init__(self, **opts)
        pattern = "([0-9][0-9]?)/([0-9][0-9]?)/([0-9][0-9]([0-9][0-9])?)"
        self.exp1 = re.compile(pattern)
        self.exp2 = re2.compile(pattern)

    def data(self, N=None, dim=None, **opts):
        # The benchmark requires a new datasets each time.
        a = ' a 01/02 5/5/9999 ' * N
        return (numpy.array([a + a.join(["01/02/2019"] * dim) for i in range(N)]), )

    def fcts(self, dim=None, **kwargs):
        # The function returns the prediction functions to tests.

        def search_re(X):
            return [self.exp1.search(x) for x in X]

        def search_re2(X):
            return [self.exp2.search(x) for x in X]

        return [{'test': 're', 'fct': search_re},
                {'test': 're2', 'fct': search_re2}]

    def validate(self, results, **kwargs):
        """
        Checks that methods *predict* and *predict_proba* returns
        the same results for both :epkg:`scikit-learn` and
        :epkg:`onnxruntime`.
        """
        res = results
        d1 = res[0][2]
        d2 = res[1][2]
        assert len(d1) == len(d2)
        for a, b in zip(d1, d2):
            g1 = a.groups()[0]
            g2 = b.groups()[0]
            assert g1 == g2


##############################
# Benchmark
# +++++++++
import pandas


def run_bench(repeat=100, verbose=False):
    pbefore = dict(dim=[2, 5, 10])
    pafter = dict(N=[1, 10, 100])

    bp = BenchPerf(pbefore, pafter, Re2Bench)

    results = list(bp.enumerate_run_benchs(repeat=repeat, verbose=verbose))
    results_df = pandas.DataFrame(results)
    return results_df


##############################
# Run the benchmark
# +++++++++++++++++


df = run_bench(verbose=True)
df.to_csv("bench_re2.csv", index=False)
print(df.head())

#########################
# Extract information about the machine used
# ++++++++++++++++++++++++++++++++++++++++++

from pymlbenchmark.context import machine_information
pkgs = ['numpy', 'pandas', 'sklearn']
dfi = pandas.DataFrame(machine_information(pkgs))
print(dfi)

#############################
# Plot the results
# ++++++++++++++++

from pymlbenchmark.plotting import plot_bench_results
print(df.columns)
plot_bench_results(df, row_cols=['N'],
                   col_cols=None, x_value='dim',
                   hue_cols=None,
                   cmp_col_values='test',
                   title="re2\nBenchmark re / re2")

import matplotlib.pyplot as plt
import sys
if "--quiet" not in sys.argv:
    plt.show()
