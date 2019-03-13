import os
import pandas
import matplotlib.pyplot as plt
from pyquickhelper.pandashelper import df2rst
from pymlbenchmark.benchmark.bench_helper import bench_pivot

name = "../../onnx/results/bench_plot_onnxruntime_random_forest.time.csv"
df = pandas.read_csv(name)
print(df2rst(df, number_format=4))
