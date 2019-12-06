
.. _l-bench-plot-onnxprofiling-decisiontreeclassifier:

Profiling DecisionTreeClassifier
================================

.. postcontents::
 
.. runpython::
    :rst:
    :sphinx: false

    import os
    import glob
    from mlprodict.tools.filename_helper import (
        extract_information_from_filename,
        make_readable_title)

    pattern = "onnx/profiles/*DecisionTreeClassifier*.svg"
    done = 0
    for name in glob.glob(pattern):
        name = name.replace("\\", "/")
        filename = os.path.splitext(os.path.split(name)[-1])[0]
        title = make_readable_title(
            extract_information_from_filename(filename))
        print(title)
        print("+" * len(title))
        print()
        print(".. raw:: html")
        print("    :file: ../../{}".format(name))
        print()
        done += 1
    if done == 0:
        print("No file found.", os.path.abspath("onnx/profiles"))
