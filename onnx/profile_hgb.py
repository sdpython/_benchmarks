import time
import numpy as np
from joblib import Memory
import sklearn
from sklearn.datasets import make_regression
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from skl2onnx import to_onnx
from mlprodict.onnxrt import OnnxInference

m = Memory(location="c:\\temp", mmap_mode='r')

# @m.cache
def make_model():
    X, y = make_regression(n_features=20, n_samples=50000)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = HistGradientBoostingRegressor(max_depth=6, max_iter=50)
    model.fit(X_train, y_train)
    onx = to_onnx(model, X.astype(np.float32))
    onxb = onx.SerializeToString()
    return model, X_test, onxb

model, X, onxb = make_model()
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
