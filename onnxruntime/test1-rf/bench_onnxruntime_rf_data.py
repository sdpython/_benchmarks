"""
Trains models and converts them into onnx.
"""
import pickle
import numpy
from tqdm import tqdm
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from mlprodict.onnxrt.onnx2py_helper import to_bytes
from skl2onnx import to_onnx


def get_data(n=50000, nf=20, **kwargs):
    X, y = make_regression(n, nf)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, **kwargs):
    model = RandomForestRegressor(**kwargs)
    model.fit(X_train, y_train)
    return model


def main():
    X_train, X_test, y_train, y_test = get_data()
    
    fullname = "data-%d-nf%d.pkl" % X_train.shape
    pkl = dict(X_test=X_test)
    with open(fullname, "wb") as f:
        pickle.dump(pkl, f)

    fullname = "data-%d-nf%d.pb" % X_train.shape
    pb = to_bytes(X_test.astype(numpy.float32))
    with open(fullname, "wb") as f:
        f.write(pb)

    fullname = "data-1-nf%d.pb" % X_train.shape[1]
    pb = to_bytes(X_test[:1].astype(numpy.float32))
    with open(fullname, "wb") as f:
        f.write(pb)

    for depth in tqdm([2, 4, 6, 8, 10]):
        model = train_model(
            X_train, y_train,
            n_estimators=200, max_depth=depth)
        
        name = model.__class__.__name__
        fullname = '%s-d%02d-nf%d.pkl' % (name, depth, X_train.shape[1])
        with open(fullname, 'wb') as f:
            pickle.dump(model, f)
        
        onx = to_onnx(model, X_train[:1])
        onxb = onx.SerializeToString()
        fullname = '%s-d%02d-nf%d.onnx' % (name, depth, X_train.shape[1])
        with open(fullname, 'wb') as f:
            f.write(onxb)
        

main()
