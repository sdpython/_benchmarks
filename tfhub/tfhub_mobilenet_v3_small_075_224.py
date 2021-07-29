# SPDX-License-Identifier: Apache-2.0
print("[begin]")
import os
import numpy
print("[import]")
from _tools import generate_random_images, benchmark


def main(opset=13):
    print("[main]")
    url = "https://tfhub.dev/google/imagenet/mobilenet_v3_small_075_224/feature_vector/5?tf-hub-format=compressed"
    dest = "tf-mobilenet-v3-small-075-224"
    name = "mobilenet-v3-small-075-224"
    onnx_name = os.path.join(dest, "%s-%d.onnx" % (name, opset))

    imgs = generate_random_images(shape=(1, 224, 224, 3), scale=1.)

    print("[benchmark]")
    benchmark(url, dest, onnx_name, opset, imgs)
    print("[end]")


if __name__ == "__main__":
    main()
