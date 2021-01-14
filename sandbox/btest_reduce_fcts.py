import pickle
import numpy as np
from pandas import DataFrame
from tqdm import tqdm
import onnx
from onnx import *
import onnxruntime as ort
import time
import tensorflow as tf


cases_pf = [
    # x_shape y_shape, axes
    [[5, 5, 2], [-1, -1, -1], [0]],
    [[5, 5, 2], [-1, -1, -1], [1]],
    [[5, 5, 2], [-1, -1, -1], [2]],
    [[5, 5, 2], [-1, -1, -1], [0, 2]],
    [[5, 5, 2], [-1, -1, -1], [1, 2]],
    [[5, 5, 2], [-1, -1, -1], [0, 1, 2]],
    [[5, 5, 2], [-1, -1, -1], [0, 1]],

    [[1000, 50, 20, 10], [-1, -1, -1, -1], [0]],
    [[1000, 50, 20, 10], [-1, -1, -1, -1], [1]],
    [[1000, 50, 20, 10], [-1, -1, -1, -1], [2]],
    [[1000, 50, 20, 10], [-1, -1, -1, -1], [3]],
    [[1000, 50, 20, 10], [-1, -1, -1, -1], [0, 3]],
    [[1000, 50, 20, 10], [-1, -1, -1, -1], [1, 2]],
    [[1000, 50, 20, 10], [-1, -1, -1, -1], [2, 3]],
    [[1000, 50, 20, 10], [-1, -1, -1, -1], [1, 2, 3]],
    [[1000, 50, 20, 10], [-1, -1, -1, -1], [0, 1, 2, 3]],
    [[1000, 50, 20, 10], [-1, -1, -1, -1], [0, 1, 2]],
    [[1000, 50, 20, 10], [-1, -1, -1, -1], [0, 1]],

    [[100, 200, 4096], [-1, -1, -1], [0]],
    [[100, 200, 4096], [-1, -1, -1], [1]],
    [[100, 200, 4096], [-1, -1, -1], [2]],
    [[100, 200, 4096], [-1, -1, -1], [0, 2]],
    [[100, 200, 4096], [-1, -1, -1], [1, 2]],
    [[100, 200, 4096], [-1, -1, -1], [0, 1, 2]],
    [[100, 200, 4096], [-1, -1, -1], [0, 1]],

    [[10, 10, 512, 512], [-1, -1, -1, -1], [0]],
    [[10, 10, 512, 512], [-1, -1, -1, -1], [1]],
    [[10, 10, 512, 512], [-1, -1, -1, -1], [2]],
    [[10, 10, 512, 512], [-1, -1, -1, -1], [3]],
    [[10, 10, 512, 512], [-1, -1, -1, -1], [1, 2]],
    [[10, 10, 512, 512], [-1, -1, -1, -1], [0, 3]],
    [[10, 10, 512, 512], [-1, -1, -1, -1], [2, 3]],
    [[10, 10, 512, 512], [-1, -1, -1, -1], [1, 2, 3]],
    [[10, 10, 512, 512], [-1, -1, -1, -1], [0, 1, 2, 3]],
    [[10, 10, 512, 512], [-1, -1, -1, -1], [0, 1, 2]],
    [[10, 10, 512, 512], [-1, -1, -1, -1], [0, 1]],

    [[2, 2, 256, 256, 256], [-1, -1, -1, -1, -1], [0]],
    [[2, 2, 256, 256, 256], [-1, -1, -1, -1, -1], [1]],
    [[2, 2, 256, 256, 256], [-1, -1, -1, -1, -1], [2]],
    [[2, 2, 256, 256, 256], [-1, -1, -1, -1, -1], [3]],
    [[2, 2, 256, 256, 256], [-1, -1, -1, -1, -1], [4]],
    [[2, 2, 256, 256, 256], [-1, -1, -1, -1, -1], [1, 3]],
    [[2, 2, 256, 256, 256], [-1, -1, -1, -1, -1], [0, 4]],
    [[2, 2, 256, 256, 256], [-1, -1, -1, -1, -1], [3, 4]],
    [[2, 2, 256, 256, 256], [-1, -1, -1, -1, -1], [2, 3, 4]],
    [[2, 2, 256, 256, 256], [-1, -1, -1, -1, -1], [1, 2, 3, 4]],
    [[2, 2, 256, 256, 256], [-1, -1, -1, -1, -1], [0, 1, 2, 3, 4]],
    [[2, 2, 256, 256, 256], [-1, -1, -1, -1, -1], [0, 1, 2, 3]],
    [[2, 2, 256, 256, 256], [-1, -1, -1, -1, -1], [0, 1, 2]],
    [[2, 2, 256, 256, 256], [-1, -1, -1, -1, -1], [0, 1]],
]


def test(x, x_shape, y_shape, axes, op='ReduceSum', repeat=100):

    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, x_shape)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, y_shape)
    max_pool = helper.make_node(op, ['X'], ['Y'], axes=axes, keepdims=1)
    graph = helper.make_graph([max_pool], 'graph', [X], [Y])
    model = helper.make_model(graph, producer_name='model')
    model.opset_import[0].version = 12
    sess = ort.InferenceSession(model.SerializeToString())
    start_at = time.clock()
    for i in range(0, repeat):
        y = sess.run(['Y'], {'X': x})
    return y[0], time.clock() - start_at


def get_tf_reduce_sum(Xtf, axes, repeat=100):
    start_at = time.clock()
    for i in range(0, repeat):
        y = tf.math.reduce_sum(Xtf, axis=axes, keepdims=True)
    t = time.clock() - start_at
    ny = y.numpy()
    return ny, t


all_pass = True
obs = []

for op in ['ReduceMax', 'ArgMax', 'ReduceLogSumExp',
           'ReduceSum', 'ReduceMean', 'ReduceMin',
           'ReduceL1', 'ReduceL2',
           'ReduceProd', 'ReduceSumSquare',
           'ArgMin',
           'ReduceLogSum', 'ReduceLogSumExp'][:3]:
    print('**', op)
    for i, case in enumerate(tqdm(cases_pf)):
        # if i != 30: continue
        x = np.random.uniform(size=case[0]).astype(np.float32)
        ort_y, lasped = test(x, case[0], case[1], case[2], op=op)
        tf_y, lasped_tf = get_tf_reduce_sum(tf.convert_to_tensor(x), case[2])
        o = dict(case=str(i), shape=case[0], axes=case[2],
                 ort=lasped, tf=lasped_tf, size=x.size,
                 ratio=lasped / lasped_tf, op=op)
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
if len(obs) > 10:
    print(df)
    df.to_csv("reduce_bench.csv", index=False)
    df.to_excel("reduce_bench.xlsx", index=False)
else:
    print(df.T)

if all_pass:
    print("All cases passed!")
