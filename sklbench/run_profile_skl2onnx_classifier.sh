echo --CREATE-FOLDER--
mkdir profiles
cd profiles

echo --BENCH-CREATE--
python -m mlprodict asv_bench --location . -n "4,5" -d "1,1000" -o -1 --add_pyspy 1 --runtime "scikit-learn,python,onnxruntime1" --conf_params "project,asv-skl2onnx;project_url,https://github.com/sdpython/asv-skl2onnx" --models SVC,RandomForestClassifier,DecisionTreeClassifier,AdaBoostClassifier,LogisticRegression,KNeighborsClassifier,MLPClassifier,MultinomialNB,BernoulliNB,OneVsRestClassifier -v 1 || exit 1

echo --PROFILE-RUN--

# bash ./pyspy/ensemble/AdaBoostClassifier/

echo --RandomForestClassifier--
cd ./pyspy/ensemble/RandomForestClassifier
export PYTHONPATH=../../../benches/ensemble/RandomForestClassifier
for f in ./*.sh;
echo -- $f --
bash $f || exit 1;
done
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


