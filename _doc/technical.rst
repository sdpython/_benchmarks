=================
Technical details
=================

Benchmarking is an exact science as the results
may change depending on the machine used to compute
the figures. There is not necessarily an exact correlation
between the processing time and the algorithm cost.
The results may also depend on the options used
to compile a library (CPU, GPU, MKL, ...).
Next sections gives some details on how it was done.

scikit-learn
============

:epkg:`scikit-learn` is usually the current latest
stable version except if the test involves a pull request
which implies :epkg:`scikit-learn` is installed from
the master branch.

onnxruntime
===========

:epkg:`onnxruntime` is not easy to install on Linux even on CPU.
The current implementation requires that :epkg:`Python` is built
with a specific flags ``--enable-shared``:

::

    ./configure --enable-optimizations --with-ensurepip=install --enable-shared --prefix=/opt/bin

This is due to a feature which requests to be able to interpret
*Python* inside a package itself and more specifically: `Embedding the Python interpreter
<https://pybind11.readthedocs.io/en/stable/compiling.html#embedding-the-python-interpreter>`_.
Then the environment variable ``LD_LIBRARY_PATH`` must be set to
the location of the shard libraries, ``/opt/bin`` in the previous example.
The following issue might appear:

::

    UserWarning: Cannot load onnxruntime.capi.
    Error: 'libnnvm_compiler.so: cannot open shared object file: No such file or directory'
