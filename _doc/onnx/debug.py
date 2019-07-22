
def label_fct(la):
    la = la.replace("onxpython", "opy")
    la = la.replace("onxonnxruntime1", "ort")
    la = la.replace("True", "1")
    la = la.replace("False", "0")
    la = la.replace("max_depth", "mxd")
    la = la.replace("method=predict_proba", "prob")
    la = la.replace("method=predict", "cl")
    return la


import matplotlib.pyplot as plt
import pandas
from pymlbenchmark.benchmark.bench_helper import bench_pivot
from pymlbenchmark.plotting import plot_bench_xtime

name = "../../onnx/results/bench_plot_onnxruntime_random_forest.perf.csv"
df = pandas.read_csv(name)
print(df.head().T)

plot_bench_xtime(df, row_cols='N', col_cols='method',
                 hue_cols=['n_estimators'],
                 cmp_col_values=('lib', 'skl'),
                 x_value='mean', y_value='xtime',
                 parallel=(1., 0.5), title=None,
                 ax=None, box_side=4, label_fct=label_fct)
plt.suptitle(
    "Acceleration onnxruntime / scikit-learn for RandomForestClassifier")
plt.show()
