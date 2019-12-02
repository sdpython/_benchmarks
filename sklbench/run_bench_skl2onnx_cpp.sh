echo --PIP--
pip install git+https://github.com/sdpython/asv.git@jenkins
pip install git+https://github.com/scikit-learn/scikit-learn.git

echo --CLONE--
if [ ! -d asv-skl2onnx ]
then
    echo --CLONE--
    git clone https://github.com/sdpython/asv-skl2onnx.git --recursive
else
    echo --UPDATE--
    cd asv-skl2onnx
    git pull
    git submodule update --init --recursive
    cd ..
fi
-n
cd asv-skl2onnx
git pull
echo --BENCH-CREATE--
python3.7 -m mlprodict asv_bench --location . -n 4,20,100 --runtime "scikit-learn,python,onnxruntime1" --conf_params "project,asv-skl2onnx;project_url,https://github.com/sdpython/asv-skl2onnx" --models SVR,SVC,NuSVC,NuSVR,LinearSVC,LinearSVR,RandomForestRegressor,RandomForestClassifier,DecisionTreeRegressor,DecisionTreeClassifier,AdaBoostClassifier,AdaBoostRegressor,LinearRegression,LogisticRegression -v 1 || exit 1
echo --BENCH-RUN--
python3.7 -m asv run --show-stderr --config asv.conf.json
if [ -d html ]
then
    echo --REMOVE HTML--
    rm html -r -f
fi
echo --PUBLISH--
python3.7 -m asv publish --config asv.conf.json -o html || exit 1
python3.7 -m mlprodict asv2csv -f results -o "asv_<date>.csv" --baseline skl || exit 1
