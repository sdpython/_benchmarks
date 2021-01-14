import pickle
import numpy as np
from pandas import DataFrame
from tqdm import tqdm
import onnx
from onnx.numpy_helper import from_array
from onnx import *
import onnxruntime as ort
import time
import tensorflow as tf


cases_pf = [
    # x_shape y_shape, axes
    [[5, 5, 2], [-1, -1, -1], [1, 1, 1, 1, 1, 1]],
    [[5, 5, 2], [-1, -1, -1], [2, 2, 2, 2, 2, 2]],

    [[1000, 50, 20, 10], [-1, -1, -1, -1], [1, 1, 1, 1, 1, 1, 1, 1]],
    [[1000, 50, 20, 10], [-1, -1, -1, -1], [2, 2, 2, 2, 2, 2, 2, 2]],

    [[100, 200, 4096], [-1, -1, -1], [1, 1, 1, 1, 1, 1]],
    [[100, 200, 4096], [-1, -1, -1], [2, 2, 2, 2, 2, 2]],

    [[10, 10, 512, 512], [-1, -1, -1, -1], [1, 1, 1, 1, 1, 1, 1, 1]],
    [[10, 10, 512, 512], [-1, -1, -1, -1], [2, 2, 2, 2, 2, 2, 2, 2]],

    [[2, 2, 256, 256, 256], [-1, -1, -1, -1, -1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
    [[2, 2, 256, 256, 256], [-1, -1, -1, -1, -1], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]],
]


def test(x, x_shape, y_shape, pads, op='Pad', repeat=100):

    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, x_shape)
    P = helper.make_tensor_value_info('P', TensorProto.INT64, [len(pads), ])
    C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [1, ])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, y_shape)
    npads = np.array(pads, dtype=np.int64)
    cst = np.array([5.5], dtype=np.float32)
    ops = [helper.make_node(op, ['X', 'P', 'C'], ['Yt']),
           helper.make_node(op, ['Yt', 'P', 'C'], ['Ytt']),
           helper.make_node(op, ['Ytt', 'P', 'C'], ['Y'])]
    graph = helper.make_graph(ops, 'graph', [X, P, C], [Y])
    model = helper.make_model(graph, producer_name='model')

    sess = ort.InferenceSession(model.SerializeToString())
    start_at = time.perf_counter()
    for i in range(0, repeat):
        y = sess.run(['Y'], {'X': x, 'P': npads, 'C': cst})
    return y[0], time.perf_counter() - start_at


def get_tf_pad(Xtf, raw_pads, repeat=100):
    half = raw_pads.shape[0] // 2
    paddings = tuple((raw_pads[i], raw_pads[i + half])
                     for i in range(0, half))
    tfpad = tf.constant(paddings)
    start_at = time.perf_counter()
    for i in range(0, repeat):
        y = tf.pad(
            tf.pad(
                tf.pad(Xtf, tfpad, constant_values=5.5),
                tfpad, constant_values=5.5),
            tfpad, constant_values=5.5)
    t = time.perf_counter() - start_at
    ny = y.numpy()
    return ny, t


def get_numpy_pad(X, raw_pads, repeat=100):
    half = raw_pads.shape[0] // 2
    paddings = tuple((raw_pads[i], raw_pads[i + half])
                     for i in range(0, half))
    start_at = time.perf_counter()
    for i in range(0, repeat):
        y = np.pad(
            np.pad(
                np.pad(X, paddings, constant_values=5.5),
                paddings, constant_values=5.5),
            paddings, constant_values=5.5)
    t = time.perf_counter() - start_at
    return y, t


all_pass = True
obs = []

for op in ['Pad']:
    print('**', op)
    for i, case in enumerate(tqdm(cases_pf)):
        # if i != 30: continue
        x = np.random.uniform(size=case[0]).astype(np.float32)
        ort_y, lasped = test(x, case[0], case[1], case[2], op=op)
        tf_y, lasped_tf = get_tf_pad(
            tf.convert_to_tensor(x), np.array(case[2]))
        np_y, lasped_np = get_numpy_pad(x, np.array(case[2]))
        o = dict(case=str(i), shape=case[0], paddings=case[2],
                 ort=lasped, tf=lasped_tf, size=x.size,
                 ratio_tf=lasped / lasped_tf, op=op,
                 np=lasped_tf, ratio_np=lasped / lasped_np)
        if not np.allclose(ort_y, tf_y):
            all_pass = False
            print("Mismatch on", case)
            o['mismatch'] = np.max(np.abs(ort_y.ravel() - tf_y.ravel()))
            filename = "error-%s-%d.pkl" % (op, i)
            with open(filename, "wb") as f:
                pickle.dump(
                    dict(x=x, ort_y=ort_y, tf_y=tf_y, err=o['mismatch']), f)
            # break
        obs.append(o)

df = DataFrame(obs)
if len(obs) > 0:
    print(df)
    df.to_csv("pad_bench.csv", index=False)
    df.to_excel("pad_bench.xlsx", index=False)
    with open("pad_bench.md", "wb") as f:
        df.to_markdown(f, index=False)
else:
    print(df.T)

if all_pass:
    print("All cases passed!")
