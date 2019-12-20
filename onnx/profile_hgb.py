import time
import numpy as np
import sklearn
from sklearn.datasets import make_regression, load_diabetes
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from skl2onnx import to_onnx
from mlprodict.onnxrt import OnnxInference


def make_model(data):
    if data == "dia":
        d = load_diabetes()
        X, y = d.data, d.target
    else:
        X, y = make_regression(n_features=20, n_samples=50000)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = HistGradientBoostingRegressor(max_depth=6, max_iter=100)
    # model = GradientBoostingRegressor(max_depth=6, n_estimators=100)
    model.fit(X_train, y_train)
    onx = to_onnx(model, X.astype(np.float32))
    onxb = onx.SerializeToString()
    return model, X_test, onxb


model, X, onxb = make_model("dia")
X = X.astype(np.float32)
print(X.shape)
oinf = OnnxInference(onxb, runtime="python_compiled")

def f1_pyrt():
    for i in range(0, 500):
        y = oinf.run({'X': X[i: i+10000]})

def f2_skl():
    with sklearn.config_context(assume_finite=True):
        for i in range(0, 500):
            model.predict(X[i: i+10000])

if X.shape[0] < 12000:
    rows = []
    while len(rows) < (12000 // X.shape[0]):
        rows.append(X)
    X = np.vstack(rows)

tts = [time.time()]
print('begin')

f2_skl()
tts.append(time.time())
print(tts[-1] - tts[-2])

f1_pyrt()
tts.append(time.time())
print(tts[-1] - tts[-2])

# f2_skl()
# tts.append(time.time())
# print(tts[-1] - tts[-2])

# f1_pyrt()
# tts.append(time.time())
# print(tts[-1] - tts[-2])
