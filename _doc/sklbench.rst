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


The results can be seen at this
`location <sklbench_results/index.html>`_.
