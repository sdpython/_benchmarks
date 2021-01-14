"""
List of models.
"""
import numpy


def get_zeros_int64(shape):
    """Get zeros."""
    return numpy.zeros(shape).astype(numpy.int64)


def get_ones_int32(shape):
    """Get ones."""
    return numpy.ones(shape).astype(numpy.int32)


def get_small_rand_int32(shape):
    """Get random ints in range [1, 99]"""
    return numpy.random.randint(
        low=1, high=100, size=shape, dtype=numpy.int32)


def get_zeros_then_ones(shape):
    """Fill half the tensor with zeros and the rest with ones"""
    cnt = numpy.prod(shape)
    zeros_cnt = cnt // 2
    ones_cnt = cnt - zeros_cnt
    return numpy.concatenate((
        numpy.zeros(zeros_cnt, dtype=numpy.int32),
        numpy.ones(ones_cnt, dtype=numpy.int32))).reshape(shape)


MODELS = {
    'bert_en_cased_L-12_H-768_A-12-v2': dict(
        url='https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/2?tf-hub-format=compressed',
        inputs=["input_mask:0",
                "input_type_ids:0",
                "input_word_ids:0"],
        outputs=["Identity:0",
                 "Identity_1:0"],
        onnx_inputs={
            "input_word_ids:0": get_small_rand_int32([1, 512]),
            "input_type_ids:0": get_ones_int32([1, 512]),
            "input_mask:0": get_zeros_then_ones([1, 512]),
        },
        tag="serve",
        signature_def="serving_default",
    ),
    'bert_en_cased_L-12_H-768_A-12-v3': dict(
        url='https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3?tf-hub-format=compressed',
        inputs=["serving_default_input_mask:0",
                "serving_default_input_type_ids:0",
                "serving_default_input_word_ids:0"],
        outputs=["embeddings/layer_norm/gamma:0",
                 "embeddings/layer_norm/beta:0"],
        onnx_inputs={
            "serving_default_input_word_ids:0": get_small_rand_int32([1, 512]),
            "serving_default_input_type_ids:0": get_ones_int32([1, 512]),
            "serving_default_input_mask:0": get_zeros_then_ones([1, 512]),
        },
        tag="serve",
        signature_def="serving_default",
    ),
}
