import os
this = os.path.dirname(os.path.abspath(__file__))

for fold in ["skl2onnx", "onnx"]:
    print("#########################")
    print()
    f = os.path.join(this, "..", fold)
    names = os.listdir(f)
    for name in names:
        fullname = os.path.join(f, name)
        short, ext = os.path.splitext(name)
        if ext != '.py':
            continue
        cmd = ["python %CDIR%/{}/{} --quiet".format(fold, name),
               "copy results\\{}*.csv %CDIR%\\{}\\results".format(short, fold),
               "copy results\\{}*.xlsx %CDIR%\\{}\\results".format(short, fold)]
        print("\n".join(cmd))
        print()

    