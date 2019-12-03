=======================
scikit-learn benchmarks
=======================

:epkg:`scikit-learn`'s team has started to develop
a benchmark located here:
:epkg:`scikit-learn_benchmarks`
I replicate here the steps I used to run and publish it
from a local machine.

.. contents::
    :local:

Installation
============

I followed the steps

::

    git clone https://github.com/jeremiedbb/scikit-learn_benchmarks.git
    cd scikit-learn_benchmarks

Run a benchmark
===============

I then ran a first benchmark with my current
installation of *Python*.

::

    asv run -b LinearRegression --no-pull

The tests do not store any result with option
``--option=<python_path>``.

Publish a benchmark
===================

I then published it on a local directory.

::

    asv publish -o html

Server to display the content
=============================

I created a key

::

    openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365

And then a small application:

::

    from starlette.routing import Router, Mount
    from starlette.staticfiles import StaticFiles

    app = Router(routes=[
        Mount('/', app=StaticFiles(directory='html'), name="html"),
    ])

And a server:

::

    uvicorn webapp:app ---host <host> -port 8877 --ssl-keyfile=./key.pem --ssl-certfile=./cert.pem

Other benchmarks
================

The first benchmark extends the official :epkg:`scikit-learn`
benchmarks available at `scikit-learn_benchmarks
<https://github.com/jeremiedbb/scikit-learn_benchmarks>`_.
The results can be seen at this
`Scikit-Learn/ONNX benchmark with AirSpeedVelocity
<http://www.xavierdupre.fr/app/benches/scikit-learn_benchmarks/index.html>`_.

The second benchmark is produced using an automated way implemented in
:epkg:`mlprodict`. The sources are available at
`asv-skl2onnx <https://github.com/sdpython/asv-skl2onnx>`_ and
displayed at `Prediction with scikit-learn and ONNX benchmark
<http://www.xavierdupre.fr/app/benches/asv-skl2onnx/index.html>`_.
A subset of these models is available at
`Prediction with scikit-learn and ONNX benchmark (SVM + Trees)
<http://www.xavierdupre.fr/app/benches/asv-skl2onnx-cpp/index.html>`_.

The last benchmark is a standalone benchmark only comparing
:epkg:`onnxruntime` and :epkg:`scikit-learn`.
The sources are available at
`scikit-onnx-benchmark <https://github.com/xadupre/scikit-onnx-benchmark>`_ and
displayed at `onnxruntime vs scikit-learn for comparison
<http://www.xavierdupre.fr/app/benches/scikit-onnx-benchmark/index.html>`_.

I also created two mini benchmark to get a sense of what the previous ones
look like:
`mlprodict model of benchmark
<http://www.xavierdupre.fr/app/benches/mlprodict_bench/index.html>`_,
`mlprodict model applied to linear models
<http://www.xavierdupre.fr/app/benches/mlprodict_bench2/index.html>`_.
