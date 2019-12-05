echo --PIP--
pip install git+https://github.com/sdpython/asv.git@jenkins
pip install git+https://github.com/scikit-learn/scikit-learn.git

echo --CREATE-FOLDER--
mkdir profiles
cd profiles

echo --BENCH-CREATE--
python3.7 -m mlprodict asv_bench --location . -n 4,50 -d 1,1000 -o last --add_pyspy 1 --runtime "scikit-learn,python,onnxruntime1" --conf_params "project,asv-skl2onnx;project_url,https://github.com/sdpython/asv-skl2onnx" --models SVC,RandomForestClassifier,DecisionTreeClassifier,AdaBoostClassifier,LogisticRegression,KNeighborsClassifier,MLPClassifier,MultinomialNB,BernoulliNB,OneVsRestClassifier -v 1 || exit 1

echo --PROFILE-RUN--

# bash ./pyspy/ensemble/AdaBoostClassifier/

bash ./pyspy/ensemble/RandomForestClassifier/bench_RandomForestClassifier_default_b_cl_1_4_12_float_.sh
bash ./pyspy/ensemble/RandomForestClassifier/bench_RandomForestClassifier_default_b_cl_1_50_12_float_.sh
bash ./pyspy/ensemble/RandomForestClassifier/bench_RandomForestClassifier_default_b_cl_1000_4_12_float_.sh
bash ./pyspy/ensemble/RandomForestClassifier/bench_RandomForestClassifier_default_b_cl_1000_50_12_float_.sh

# bash ./pyspy/linear_model/LogisticRegression/ nozipmap raw_score
# bash ./pyspy/naive_bayes/BernoulliNB/
# bash ./pyspy/naive_bayes/MultinomialNB/
# bash ./pyspy/neighbors/KNeighborsClassifier/ cdist
# bash ./pyspy/neural_network/MLPClassifier/
# bash ./pyspy/svm/SVC/ linear poly rbf sigmoid
# bash ./pyspy/tree/LinearSVC/DecisionTreeClassifier

echo --PUBLISH--
mkdir svg
