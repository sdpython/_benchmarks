
.. _l-bench-plot-onnxprofiling-gradientboostingregressor:

Profiling GradientBoostingRegressor
===================================

The same model is measured through the following profilings,
they depend on the following parameters.

* *problem*: see :epkg:`find_suitable_problem`
* *scenario*: see :epkg:`build_custom_scenarios`
* *N*: batch size
* *nf*: number of features
* *ops*: opset
* anything else: options
* *by line* or *by fct*: profile show either
  line number either function names

.. postcontents::

.. runpython::
    :rst:
    :sphinx: false

    import os
    import glob
    from mlprodict.tools.filename_helper import (
        extract_information_from_filename,
        make_readable_title)

    pattern = "onnx/profiles_reg/*GradientBoostingReg*.svg"
    done = 0
    pubs = []
    for name in glob.glob(pattern):
        name = name.replace("\\", "/")
        filename = os.path.splitext(os.path.split(name)[-1])[0]
        title = make_readable_title(
            extract_information_from_filename(filename))
        pubs.append((title, filename, name))
    pubs.sort()
    for title, filename, name in pubs:
        print(title)
        print("+" * len(title))
        print()
        print(".. raw:: html")
        print("    :file: ../../{}".format(name))
        print()
        done += 1
    if done == 0:
        print("No file found.", os.path.abspath("onnx/profiles_reg"))
