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

onnx
====

``python setup.py install`` works on Linux. On Windows,
*protobuf* must be compiled first and then referenced
before building the wheel. This is what it could look
like:

::

    @echo off
    set ONNX_ML=1
    set PATH=c:\Python370_x64;c:\Python370_x64\Scripts;%PATH%
    set PATH=%~dp0..\\..\protobuf\build_msvc\Release;%PATH%
    set CMAKE_ARGS=..\\third_party\\pybind11\\tools -DPROTOBUF_INCLUDE_DIRS=C:\\xavierdupre\\__home_\\github_fork\\protobuf\\src -DPROTOBUF_LIBRARIES=C:\\xavierdupre\\__home_\\github_fork\\protobuf\\build_msvc\\Release\\libprotobuf.lib;C:\\xavierdupre\\__home_\\github_fork\\protobuf\\build_msvc\\Release\\libprotoc.lib -DONNX_PROTOC_EXECUTABLE=C:\\xavierdupre\\__home_\\github_fork\\protobuf\\build_msvc\\Release\\protoc.exe
    cd onnx
    python setup.py bdist_wheel
    cd ..

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

To build :epkg:`onnxruntime`:

::

    git clone https://github.com/Microsoft/onnxruntime.git --recursive
    export LD_LIBRARY_PATH=/usr/local/Python-3.6.8
    python3.6 ./onnxruntime/tools/ci_build/build.py --build_dir ./onnxruntime/build/debian36 --config Release --enable_pybind --build_wheel --use_mkldnn --use_openmp --build_shared_lib
    export LD_LIBRARY_PATH=/usr/local/Python-3.7.2
    python3.7 ./onnxruntime/tools/ci_build/build.py --build_dir ./onnxruntime/build/debian37 --config Release --enable_pybind --build_wheel --use_mkldnn --use_openmp --build_shared_lib

If the wheel then, it is possible to just copy the files
into the *python* distribution:

::

    cp -r ./onnxruntime/build/debian36/Release/onnxruntime /usr/local/lib/python3.6/site-packages/
    cp -r ./onnxruntime/build/debian37/Release/onnxruntime /usr/local/lib/python3.7/site-packages/
