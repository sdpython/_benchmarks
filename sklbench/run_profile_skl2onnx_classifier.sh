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
python3.7 -m mlprodict asv_bench --location . -n 1,1000 --add_pyspy 1 --runtime "scikit-learn,python,onnxruntime1" --conf_params "project,asv-skl2onnx;project_url,https://github.com/sdpython/asv-skl2onnx" --models SVC,NuSVC,LinearSVC,RandomForestClassifier,DecisionTreeClassifier,AdaBoostClassifier,LogisticRegression,KNeighborsClassifier,MLPClassifier,MultinomialNB,BernoulliNB,OneVsRestClassifier -v 1 || exit 1
echo --PROFILE-RUN--
if [ -d html ]
then
    echo --REMOVE HTML--
    rm html -r -f
fi
echo --PUBLISH--
