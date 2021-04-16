echo --CREATE-FOLDER--
mkdir profiles
cd profiles

echo --BENCH-CREATE--
python -m mlprodict asv_bench --location . -n "4,50" -d "1,1000" -o -1 --add_pyspy 1 --runtime "scikit-learn,python_compiled,onnxruntime1" --conf_params "project,asv-skl2onnx;project_url,https://github.com/sdpython/asv-skl2onnx" --models KNeighborsClassifier -v 1 || exit 1

echo --PROFILE-RUN--

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


echo --PUBLISH--
mkdir htmlsvg
cp -v pyspy/*/*/*.svg htmlsvg