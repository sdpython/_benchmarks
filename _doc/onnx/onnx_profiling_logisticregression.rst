
.. _l-bench-plot-onnxprofiling-logisticregression:

Profiling LogisticRegression
============================

.. runpython::
    :rst:

    import os
    import glob

    pattern = "../../onnx/profiles/*LogisticRegression*.svg"
    for name in glob.glob(pattern):
        name = name.replace("\\", "/")
        filename = os.path.splitext(os.path.split(name)[-1])[0]
        spl = filename.split('_')
        model = spl[1]
        sce = spl[2]
        problem = "_".join(spl[3:5])
        dim = spl[5]
        nf = spl[6]
        opset = spl[7]
        opt = '_'.join(spl[8:]).strip('_')
        
        title = "{model} p:{problem} s:{sce} N={N} d={d} opset={opset} opt={opt}".format(
            model=model, problem=problem, N=dim, d=nf, opt=opt, opset=opset, sce=sce)
        print(title)
        print("+" * len(title))
        print()
        print(".. raw:: html")
        print("    :file: {}".format(name))
        print()
