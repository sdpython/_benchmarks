"""
Helpers.
"""
import os
import urllib.request
import numpy
from tqdm import tqdm
from pyquickhelper.filehelper.compression_helper import (
    ungzip_files, untar_files)
from pyquickhelper.loghelper import run_cmd
from models import MODELS


LOCATION = os.path.join(os.path.abspath(os.path.dirname(__file__)), "_models")


def name_version(url):
    """
    Returns a model name and version assuming the model
    was downloaded from *tfhub*.

    Example:

    ::

        https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3?tf-hub-format=compressed
                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^
                                         name                       version
     """
    name, v = url.split('?')[0].split('/')[-2:]
    return name, v, "%s-v%s" % (name, v)


def model_exists(name, url):
    """
    Determines if a model exists.
    """
    global LOCATION
    if not os.path.exists(LOCATION):
        return None
    model, vers, fullname = name_version(url)
    fold = os.listdir(LOCATION)
    if fullname in fold:
        return os.path.join(LOCATION, name)
    return None


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path, name):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=name) as t:
        urllib.request.urlretrieve(
            url, filename=output_path, reporthook=t.update_to)


def get_node_list(pathname, verbose=0):
    """
    Returns the list of nodes in one model.
    """
    import tensorflow as tf
    if verbose:
        print("[load] %r" % pathname)
    model = tf.saved_model.load(pathname)
    all_types = []
    try:
        for op in model.graph.get_operations():
            all_types.append(op.type)
    except Exception as e:
        all_types.append(e)
    return set(all_types)


def download_model(name, url, verbose=0):
    """
    Downloads a model given its url.
    """
    loc = model_exists(name, url)
    if False and loc:
        return loc
    model, vers, fullname = name_version(url)
    fullpath = os.path.join(LOCATION, fullname)
    if not os.path.exists(fullpath):
        if verbose:
            print('[download_model] create %r.' % fullpath)
        os.makedirs(fullpath)
    outgz = os.path.join(fullpath, "model.tar.gz")
    if not os.path.exists(outgz):
        if verbose:
            print('[download_model] download from %r.' % url)
        download_url(url, outgz, fullname)
    outtar = os.path.join(fullpath, "model.tar")
    if not os.path.exists(outtar):
        if verbose:
            print('[download_model] ungzip %r.' % outgz)
        ungzip_files(outgz, fullpath, unzip=False)
    model = os.path.join(fullpath, "saved_model.pb")
    if not os.path.exists(model):
        if verbose:
            print('[download_model] untar %r.' % outtar)
        untar_files(outtar, fullpath)
    return fullpath


def convert(pathname, verbose=0):
    """
    Converts into ONNX.
    """
    # tflite
    lite = os.path.join(pathname, 'model.lite')
    if False and not os.path.exists(lite):
        import tensorflow.lite as tfl
        if verbose:
            print('[convert] to lite %r.' % pathname)
        converter = tfl.TFLiteConverter.from_saved_model(pathname)
        try:
            tflite_model = converter.convert()
        except Exception as e:
            print('[convert] lite failed due to %r.' % e)
            tflite_model = None
        if tflite_model is not None:
            if verbose:
                print('[convert] save lite %r.' % pathname)
            with open(lite, "wb") as f:
                f.write(tflite_model)

    # node list
    if verbose:
        res = get_node_list(pathname, verbose=verbose)
        print('[convert] node types: %r' % res)

    # onnx
    output = os.path.join(pathname, "model.onnx")
    lastname = pathname.replace("\\", "/").split('/')[-1]
    inputs = None
    if lastname not in MODELS:
        raise ValueError("Unknown model %r." % lastname)
    model = MODELS[lastname]
    inputs = model['inputs']
    outputs = model['outputs']
    tag = model['tag']
    sig = model['signature_def']
    onnx_inputs = model['onnx_inputs']
    if inputs is None:
        if verbose:
            print('[convert] to ONNX %r.' % pathname)
        raise NotImplementedError("Unable to convert %r." % lastname)
    inputs = ",".join(inputs)
    outputs = ",".join(outputs)
    if not os.path.exists(output):
        def noprint(*args):
            pass
        cmd = ["python", "-m", "tf2onnx.convert", "--saved-model", pathname,
               "--output", output, "--inputs", inputs, "--outputs", outputs,
               '--tag', tag, '--signature_def', sig]
        out, err = run_cmd(" ".join(cmd), wait=True,
                           fLOG=print if verbose else noprint, shell=True)
    return pathname, output, inputs, outputs, onnx_inputs


def load_model(name, verbose=0):
    """
    Name: see MODELS.
    """

    if name not in MODELS:
        raise ValueError("Unknown model %r." % name)
    url = MODELS[name]['url']
    pathname = download_model(name, url, verbose=verbose)
    return convert(pathname, verbose=verbose)


if __name__ == "__main__":
    pathname = download_model(
        "https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3?tf-hub-format=compressed")
    convert(pathname)
    # import pprint
    # pprint.pprint(get_node_list(pathname))
