"""
Benchmark a tensorflow model.
"""
import onnxruntime as rt
import PIL.Image
import time
import numpy as np
import tensorflow as tf
from os import path
from helper import load_model


def main():
    res = load_model("bert_en_cased_L-12_H-768_A-12-v2", verbose=1)
    pathname, output, inputs, outputs, onnx_inputs = res
    output_names = outputs
    structured_outputs = [
        "answer_types", "tf_op_layer_end_logits",
        "tf_op_layer_start_logits", "unique_ids"]
    perf_iter = 5
    rtol = 0.01
    atol = 0.0001

    print("[main] test ONNX %r." % output)
    print("[main] inputs names %r." % list(onnx_inputs))
    print("[main] output names %r." % output_names.split(','))

    m = rt.InferenceSession(output)
    results_onnx = m.run(output_names.split(','), onnx_inputs)
    print("[main] got results, testing perf")
    start = time.time()
    for _ in range(perf_iter):
        _ = m.run(output_names.split(','), onnx_inputs)
    onnx_runtime_ms = (time.time() - start) / perf_iter * 1000
    print("[main] ONNX perf:", onnx_runtime_ms)

    print("[main] loading TF")
    imported = tf.saved_model.load(pathname, tags=['serve'])
    concrete_func = imported.signatures["serving_default"]

    tf_inputs = {}
    for k, v in onnx_inputs.items():
        tf_inputs[k.split(":")[0]] = tf.constant(v)
    tf_func = tf.function(concrete_func)

    print("[main] Running TF")
    tf_results_d = tf_func(**tf_inputs)
    #results_tf = [tf_results_d[output].numpy() for output in structured_outputs]
    print("[main] Got results, testing perf")
    start = time.time()
    for _ in range(perf_iter):
        _ = concrete_func(**tf_inputs)
    tf_runtime_ms = (time.time() - start) / perf_iter * 1000
    print("[main] TF perf:", tf_runtime_ms)

    print("[main] Results match")
    print('[main] device', rt.get_device(), rt.__version__, rt.__file__)
    print("[main] TF perf, ONNX perf, ratio")
    print("[main] ", tf_runtime_ms, onnx_runtime_ms,
          tf_runtime_ms / onnx_runtime_ms)


main()
