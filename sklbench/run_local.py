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
        cmd = ["echo ---- RUN {}".format(name),
               "set NAME=results\\{}.csv".format(short),
               "if exists %NAME% goto next_{}:".format(short),
               "python %CDIR%/{}/{} --quiet".format(fold, name),
               "copy results\\%NAME%.csv %CDIR%\\{}\\results".format(fold),
               "copy results\\%NAME%.xlsx %CDIR%\\{}\\results".format(fold),
               ":next_{}:".format(short)]
        print("\n".join(cmd))
        print()
