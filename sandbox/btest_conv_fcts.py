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
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.algebra.onnx_ops import OnnxConv


cases_pf = [
    dict(shape_x=[1, 96, 56, 56], shape_w=[24, 96, 1, 1],
         group=1, kernel_shape=[1, 1], dilations=[1, 1],
         strides=[1, 1], padding="VALID"),
]


def test(x, w, shape_x, shape_w, group=None, kernel_shape=None,
         dilations=None, strides=None, padding=None,
         repeat=10):

    onx = OnnxConv('X', 'W', output_names=['Y'],
                   auto_pad=padding, group=group, dilations=dilations,
                   kernel_shape=kernel_shape, strides=strides,
                   op_version=12)
    model_def = onx.to_onnx({'X': x.astype(np.float32),
                             'W': w.astype(np.float32)},
                            target_opset=12,
                            outputs=[('Y', FloatTensorType())])

    sess = ort.InferenceSession(model_def.SerializeToString())
    start_at = time.perf_counter()
    for i in range(0, repeat):
        y = sess.run(['Y'], {'X': x, 'W': w})
    return y[0], time.perf_counter() - start_at


def get_tf_conv(x, w, group=None, kernel_shape=None,
                dilations=None, strides=None, padding=None,
                repeat=10):
    xi = tf.constant(x, dtype=tf.float32)
    kernel = tf.constant(w, dtype=tf.float32)
    start_at = time.perf_counter()
    for i in range(0, repeat):
        y = tf.nn.conv2d(xi, kernel, strides=strides, dilations=dilations,
                         padding=padding)
    t = time.perf_counter() - start_at
    ny = y.numpy()
    return ny, t


all_pass = True
obs = []

for i, case in enumerate(tqdm(cases_pf)):
    # if i != 30: continue
    x = np.random.uniform(size=case['shape_x']).astype(np.float32)
    w = np.random.uniform(size=case['shape_w']).astype(np.float32)
    ort_y, lasped = test(
        x, w, shape_x=x.shape, shape_w=w.shape, group=case['group'],
        kernel_shape=case['kernel_shape'], dilations=case['dilations'],
        strides=case['strides'], padding=case['padding'])
    tf_y, lasped_tf = get_tf_conv(
        tf.convert_to_tensor(x), tf.convert_to_tensor(w),
        group=case['group'], kernel_shape=case['kernel_shape'],
        dilations=case['dilations'], strides=case['strides'],
        padding=case['padding'])
    o = dict(case=str(i), ort=lasped, tf=lasped_tf, size=x.size,
             ratio_tf=lasped / lasped_tf, op=op)
    o.update(case)
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
