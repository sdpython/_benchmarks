echo --CREATE-FOLDER--
mkdir profiles
cd profiles

echo --BENCH-CREATE--
python -m mlprodict asv_bench --location . -n "4,50" -d "1,1000" -o -1 --add_pyspy 1 --runtime "scikit-learn,python,onnxruntime1" --conf_params "project,asv-skl2onnx;project_url,https://github.com/sdpython/asv-skl2onnx" --models SVC,RandomForestClassifier,DecisionTreeClassifier,AdaBoostClassifier,LogisticRegression,KNeighborsClassifier,MLPClassifier,MultinomialNB,BernoulliNB,OneVsRestClassifier -v 1 || exit 1

echo --PROFILE-RUN--

# bash ./pyspy/ensemble/AdaBoostClassifier/

cd ./pyspy/ensemble/RandomForestClassifier
export PYTHONPATH=../../../benches/linear_model/LogisticRegression
echo --CHECK--
ls ../../..
ls ../../../benches
ls ../../../benches/linear_model
ls ../../../benches/linear_model/LogisticRegression
echo --ENDCHECK--
bash ./bench_RandomForestClassifier_default_b_cl_1_4_12_float_.sh || exit 1
bash ./bench_RandomForestClassifier_default_b_cl_1_50_12_float_.sh || exit 1
bash ./bench_RandomForestClassifier_default_b_cl_1000_4_12_float_.sh || exit 1
bash ./bench_RandomForestClassifier_default_b_cl_1000_50_12_float_.sh || exit 1
cd ../../..

# bash ./pyspy/linear_model/LogisticRegression/ nozipmap raw_score
# bash ./pyspy/naive_bayes/BernoulliNB/
# bash ./pyspy/naive_bayes/MultinomialNB/
# bash ./pyspy/neighbors/KNeighborsClassifier/ cdist
# bash ./pyspy/neural_network/MLPClassifier/
# bash ./pyspy/svm/SVC/ linear poly rbf sigmoid
# bash ./pyspy/tree/LinearSVC/DecisionTreeClassifier

echo --PUBLISH--
mkdir htmlsvg
cp -v pyspy/*/*/*.svg htmlsvg


