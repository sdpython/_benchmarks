echo --CREATE-FOLDER--
mkdir profiles
cd profiles

echo --BENCH-CREATE--
python -m mlprodict asv_bench --location . -n "4,50" -d "1,1000" -o -1 --add_pyspy 1 --runtime "scikit-learn,python_compiled,onnxruntime1" --conf_params "project,asv-skl2onnx;project_url,https://github.com/sdpython/asv-skl2onnx" --models SVR,RandomForestRegressor,DecisionTreeRegressor,AdaBoostRegressor,LinearRegression,KNeighborsRegressor,MLPRegressor -v 1 || exit 1

echo --PROFILE-RUN--

echo --AdaBoostRegressor--
cd ./pyspy/ensemble/AdaBoostRegressor
export PYTHONPATH=../../../benches/ensemble/AdaBoostRegressor
for f in ./*float*.sh
do
    if [[ $f != *64* ]]
    then
        echo "run '$f'"
        bash $f
    fi
done
cd ../../..

echo --RandomForestRegressor--
cd ./pyspy/ensemble/RandomForestRegressor
export PYTHONPATH=../../../benches/ensemble/RandomForestRegressor
for f in ./*float*.sh
do
    if [[ $f != *64* ]]
    then
        echo "run '$f'"
        bash $f
    fi
done
cd ../../..

echo --DecisionTreeRegressor--
cd ./pyspy/tree/DecisionTreeRegressor
export PYTHONPATH=../../../benches/tree/DecisionTreeRegressor
for f in ./*float*.sh
do
    if [[ $f != *64* ]]
    then
        echo "run '$f'"
        bash $f
    fi
done
cd ../../..

echo --LinearRegression--
cd ./pyspy/linear_model/LinearRegression
export PYTHONPATH=../../../benches/linear_model/LinearRegression
for f in ./*float*.sh
do
    if [[ $f != *64* ]]
    then
        echo "run '$f'"
        bash $f
    fi
done
cd ../../..

echo --KNeighborsRegressor--
cd ./pyspy/neighbors/KNeighborsRegressor
export PYTHONPATH=../../../benches/neighbors/KNeighborsRegressor
for f in ./*float*.sh
do
    if [[ $f != *64* ]]
    then
        echo "run '$f'"
        bash $f
    fi
done
cd ../../..

echo --MLPRegressor--
cd ./pyspy/neural_network/MLPRegressor
export PYTHONPATH=../../../benches/neural_network/MLPRegressor
for f in ./*float*.sh
do
    if [[ $f != *64* ]]
    then
        echo "run '$f'"
        bash $f
    fi
done
cd ../../..

echo --SVR--
cd ./pyspy/svm/SVR
export PYTHONPATH=../../../benches/svm/SVR
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


