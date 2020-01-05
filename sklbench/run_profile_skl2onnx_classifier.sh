echo --CREATE-FOLDER--
mkdir profiles
cd profiles

echo --BENCH-CREATE--
python -m mlprodict asv_bench --location . -n "4,50" -d "1,1000" -o -1 --add_pyspy 1 --runtime "scikit-learn,python_compiled,onnxruntime1" --conf_params "project,asv-skl2onnx;project_url,https://github.com/sdpython/asv-skl2onnx" --models SVC,RandomForestClassifier,DecisionTreeClassifier,GradientBoostingClassifier,AdaBoostClassifier,LogisticRegression,KNeighborsClassifier,MLPClassifier,MultinomialNB,BernoulliNB,OneVsRestClassifier -v 1 || exit 1

echo --PROFILE-RUN--

echo --AdaBoostClassifier--
cd ./pyspy/ensemble/AdaBoostClassifier
export PYTHONPATH=../../../benches/ensemble/AdaBoostClassifier
for f in ./*float*.sh
do
    if [[ $f != *64* ]]
    then
        echo "run '$f'"
        bash $f
    fi
done
cd ../../..

echo --GradientBoostingClassifier--
cd ./pyspy/ensemble/GradientBoostingClassifier
export PYTHONPATH=../../../benches/ensemble/GradientBoostingClassifier
for f in ./*float*.sh
do
    if [[ $f != *64* ]]
    then
        echo "run '$f'"
        bash $f
    fi
done
cd ../../..

echo --RandomForestClassifier--
cd ./pyspy/ensemble/RandomForestClassifier
export PYTHONPATH=../../../benches/ensemble/RandomForestClassifier
for f in ./*float*.sh
do
    if [[ $f != *64* ]]
    then
        echo "run '$f'"
        bash $f
    fi
done
cd ../../..

echo --DecisionTreeClassifier--
cd ./pyspy/tree/DecisionTreeClassifier
export PYTHONPATH=../../../benches/tree/DecisionTreeClassifier
for f in ./*float*.sh
do
    if [[ $f != *64* ]]
    then
        echo "run '$f'"
        bash $f
    fi
done
cd ../../..

echo --LogisticRegression--
cd ./pyspy/linear_model/LogisticRegression
export PYTHONPATH=../../../benches/linear_model/LogisticRegression
for f in ./*float*.sh
do
    if [[ $f != *64* ]]
    then
        echo "run '$f'"
        bash $f
    fi
done
cd ../../..

echo --BernoulliNB--
cd ./pyspy/naive_bayes/BernoulliNB
export PYTHONPATH=../../../benches/naive_bayes/BernoulliNB
for f in ./*float*.sh
do
    if [[ $f != *64* ]]
    then
        echo "run '$f'"
        bash $f
    fi
done
cd ../../..

echo --MultinomialNB--
cd ./pyspy/naive_bayes/MultinomialNB
export PYTHONPATH=../../../benches/naive_bayes/MultinomialNB
for f in ./*float*.sh
do
    if [[ $f != *64* ]]
    then
        echo "run '$f'"
        bash $f
    fi
done
cd ../../..

echo --KNeighborsClassifier--
cd ./pyspy/neighbors/KNeighborsClassifier
export PYTHONPATH=../../../benches/neighbors/KNeighborsClassifier
for f in ./*float*.sh
do
    if [[ $f != *64* ]]
    then
        echo "run '$f'"
        bash $f
    fi
done
cd ../../..

echo --MLPClassifier--
cd ./pyspy/neural_network/MLPClassifier
export PYTHONPATH=../../../benches/neural_network/MLPClassifier
for f in ./*float*.sh
do
    if [[ $f != *64* ]]
    then
        echo "run '$f'"
        bash $f
    fi
done
cd ../../..

echo --SVC--
cd ./pyspy/svm/SVC
export PYTHONPATH=../../../benches/svm/SVC
for f in ./*float*.sh
do
    if [[ $f != *64* ]]
    then
        echo "run '$f'"
        bash $f
    fi
done
cd ../../..


echo --PUBLISH--
mkdir htmlsvg
cp -v pyspy/*/*/*.svg htmlsvg


