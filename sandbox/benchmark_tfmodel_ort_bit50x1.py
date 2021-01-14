import time
import numpy
import pandas
from tqdm import tqdm
import onnxruntime as ort


def generate_random_images(shape, n=10):
    imgs = []
    for i in range(n):
        sh = (1,) + shape + (3,)
        img = numpy.clip(numpy.abs(numpy.random.randn(*sh)), 0, 1) * 255
        img = img.astype(numpy.float32)
        imgs.append(img)
    return imgs


def measure_time(fct, imgs):
    results = []
    times = []
    for img in tqdm(imgs):
        begin = time.perf_counter()
        result = fct(img)
        end = time.perf_counter()
        results.append(result)
        times.append(end - begin)
    return results, times


# Download model from https://tfhub.dev/google/bit/m-r50x1/1
# python -m tf2onnx.convert --saved-model "bit_m-r50x1_1" --output bit50x1.onnx --opset 12
ort = ort.InferenceSession('bit50x1.onnx')
name = ort.get_inputs()[0].name
shape = ort.get_inputs()[0].shape
typ = ort.get_inputs()[0].type
fct_ort = lambda img: ort.run(None, {name: img})

imgs = generate_random_images((32, 32), 2)

from mlprodict.onnxrt import OnnxInference
# oinf = OnnxInference('bit50x1.onnx', runtime='python')
# oinf.run({name: imgs[0]})

print(name, typ, shape, imgs[0].shape)
ort.run(None, {name: imgs[0]})


import tensorflow as tf
import tensorflow_hub as hub
model = tf.keras.models.load_model("bit_m-r50x1_1")

res = []
for shape in [(64, 64), (128, 128), (256, 256), (512, 512)]:
    imgs = generate_random_images(shape)
    results_ort, duration_ort = measure_time(fct_ort, imgs)
    results_tf, duration_tf = measure_time(model, imgs)

    print('------------------------')
    print('shape', shape)
    print(len(imgs), duration_ort)
    print(len(imgs), duration_tf)
    print("ratio ORT / TF", sum(duration_ort) / sum(duration_tf))
    res.append(dict(shape=shape, ort=sum(duration_ort), tf=sum(duration_tf),
                    ort1=duration_ort[0], tf1=duration_tf[0],
                    max_ort=max(duration_ort), max_tf=max(duration_tf)))

df = pandas.DataFrame(res)
df['ratio'] = df['ort'] / df['tf']
print(df.T)
df.to_csv('bit50x1.bench.csv', index=False)


"""
shape     (64, 64)  (128, 128)  (256, 256)  (512, 512)
ort       0.384306      1.1959     4.46996      19.146
tf         3.54564     3.89299     5.54174     13.6423
ort1     0.0425563    0.114432    0.444982     1.96204
tf1        1.22996     1.25792     1.38657     2.20639
max_ort  0.0439235    0.135417    0.478989     1.96204
max_tf     1.22996     1.25792     1.38657     2.20639
ratio     0.108389    0.307192    0.806598     1.40343
"""

"""
shape     (64, 64)  (128, 128)  (256, 256)  (512, 512)
ort       0.397519     1.20315     4.69151      20.186
tf          3.7017     4.02371     5.89487     14.1204
ort1     0.0614692     0.12436    0.557032     1.96682
tf1        1.49747     1.33536     1.57625     2.31108
max_ort  0.0614692    0.127332    0.557032     2.07249
max_tf     1.49747     1.33536     1.57625     2.31108
ratio     0.107388    0.299014    0.795863     1.42956
"""
