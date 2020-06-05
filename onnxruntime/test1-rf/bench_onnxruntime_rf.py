"""
Trains models and converts them into onnx.
"""
import pickle
import glob
import os
import numpy
import pandas
from tqdm import tqdm
from cpyquickhelper.numbers import measure_time
from sklearn import __version__ as skl_version, config_context
from onnxruntime import InferenceSession, __version__ as ort_version
from mlprodict import __version__ as pyrt_version
from mlprodict.onnxrt import OnnxInference


def loop_model(sess, X_test):
    for i in range(X_test.shape[0]):
        sess.run(None, {'X': X_test[i: i+1]})


def loop_model2(oinf, X_test):
    for i in range(X_test.shape[0]):
        oinf.run({'X': X_test[i: i+1]})


def test1(model, X_test, number, repeat, name, **data):
    res = measure_time(
        'model.predict(X_test)', div_by_number=True,
        number=number, repeat=repeat,
        context={'X_test': X_test, 'model': model})
    res['name'] = name
    res['runtime'] = 'skl'
    res['version'] = skl_version
    res['batch'] = "y"
    res.update(data)
    return res
    

def test2(sess, X_test, number, repeat, name, **data):
    input = {'input': X_test}
    res = measure_time(
        'sess.run(None, {"X": X_test})', div_by_number=True,
        number=number, repeat=repeat,
        context={'sess': sess, 'X_test': X_test})
    res['name'] = name
    res['runtime'] = 'onnxruntime'
    res['version'] = ort_version
    res['batch'] = "y"
    res.update(data)
    return res


def test3(sess, X_test, number, repeat, name, **data):
    res = measure_time(
        'loop_model(sess, X_test)',
        div_by_number=True,
        number=number, repeat=repeat,
        context={'sess': sess, 'X_test': X_test,
                 'loop_model': loop_model})
    res['name'] = name
    res['runtime'] = 'onnxruntime'
    res['version'] = ort_version
    res['batch'] = "n"
    res.update(data)
    return res


def test4(oinf, X_test, number, repeat, name, **data):
    input = {'input': X_test}
    res = measure_time(
        'oinf.run({"X": X_test})', div_by_number=True,
        number=number, repeat=repeat,
        context={'oinf': oinf, 'X_test': X_test})
    res['name'] = name
    res['runtime'] = 'mlprodict'
    res['version'] = pyrt_version
    res['batch'] = "y"
    res.update(data)
    return res


def test5(oinf, X_test, number, repeat, name, **data):
    res = measure_time(
        'loop_model2(oinf, X_test)',
        div_by_number=True,
        number=number, repeat=repeat,
        context={'oinf': oinf, 'X_test': X_test,
                 'loop_model2': loop_model2})
    res['name'] = name
    res['runtime'] = 'mlprodict'
    res['version'] = pyrt_version
    res['batch'] = "n"
    res.update(data)
    return res



def main(nf=20, number=1, repeat=5):
    files = glob.glob("*nf%d.pkl" % nf)
    print(files)
    
    # data
    X_test = None
    for name in files:
        if "data" not in name:
            continue
        with open(name, 'rb') as f:
            X_test = pickle.load(f)
    if X_test is None:
        raise RuntimeError("Unable to find data for nf=%d" % nf)
    X_test = X_test['X_test']

    # scikit-learn
    obs = []
    
    pbar = tqdm(files)
    for name in pbar:
        if 'data' in name:
            continue
        with open(name, 'rb') as f:
            model = pickle.load(f)
        sess = InferenceSession(name.replace('.pkl', '.onnx'))
        size = os.stat(name.replace('.pkl', '.onnx')).st_size
        skl_size = os.stat(name).st_size
        oinf = OnnxInference(name.replace('.pkl', '.onnx'),
                             runtime="python_compiled")

        # sklearn
        pbar.set_postfix(rt="skl")
        obs.append(
            test1(model, X_test, number, repeat, name, model_size=skl_size))

        # onnxruntime
        pbar.set_postfix(rt="ort")
        obs.append(
            test2(sess, X_test, number, repeat, name, model_size=size))

        # onnxruntime - single
        obs.append(
            test3(sess, X_test, number, repeat, name, model_size=size))
        
        # mlprodict
        pbar.set_postfix(rt="mlp")
        obs.append(
            test4(oinf, X_test, number, repeat, name, model_size=size))

        # onnxruntime - single
        obs.append(
            test5(oinf, X_test, number, repeat, name, model_size=size))
        
    
    df = pandas.DataFrame(obs)
    df.to_csv("result-nf%d-%s.csv" % (nf, ort_version), index=False)
    df.to_excel("result-nf%d-%s.xlsx" % (nf, ort_version), index=False)
    print(df)
    return df
        

with config_context(assume_finite=True):
    main()
