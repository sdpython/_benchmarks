import os
this = os.path.dirname(os.path.abspath(__file__))

for fold in ["skl2onnx", "onnx"]:
    print("#########################")
    print()
    f = os.path.join(this, "..", fold)
    names = os.listdir(f)
    for name in names:
        fullname = os.path.join(f, name)
        short = os.path.split(name)[0]
        cmd = ["python %CDIR%/{}/{} --quiet".format(fold, name),
               "copy {}*.csv %CDIR%\\{}\\results".format(short, fold),
               "copy {}*.xlsx %CDIR%\\{}\\results".format(short, fold)]
        print("\n".join(cmd))
        print()

    