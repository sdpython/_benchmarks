# Publishing does not seem to work for this benchmark.
# It uses the existing environment.
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

cd asv-skl2onnx
git pull
echo --BENCH-CREATE--
python -m mlprodict asv_bench --location . -o -1 --runtime "scikit-learn,python_compiled,onnxruntime1" --conf_params "project,asv-skl2onnx;project_url,https://github.com/sdpython/asv-skl2onnx" --models SVR,SVC,NuSVC,NuSVR,LinearSVC,LinearSVR,RandomForestRegressor,RandomForestClassifier,DecisionTreeRegressor,DecisionTreeClassifier,AdaBoostClassifier,AdaBoostRegressor -v 1 --env same || exit 1
echo --BENCH-RUN--
python -m asv run --show-stderr --config asv.conf.json --environment same --python same
if [ -d html ]
then
    echo --REMOVE HTML--
    rm html -r -f
fi
echo --PUBLISH--
python -m asv publish --config asv.conf.json -o html || exit 1
python -m mlprodict asv2csv -f results -o "asv_<date>.csv" --baseline skl || exit 1
