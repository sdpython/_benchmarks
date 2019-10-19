echo --CLONE--
if [ ! -d scikit-onnx-benchmark ]
then
    echo --CLONE--
    git clone https://github.com/xadupre/scikit-onnx-benchmark.git --recursive
else
    echo --UPDATE--
    cd scikit-onnx-benchmark
    git pull
    git submodule update --init --recursive
    cd ..
fi

cd scikit-onnx-benchmark
echo "BEGIN" > begin.txt

echo --BENCH--
python3.7 -m asv run --show-stderr --config asv.conf.json
echo "END" > end.txt
if [ -d html ]
then
    echo --REMOVE HTML--
    rm html -r -f
fi
echo --PUBLISH--
python3.7 -m asv publish --config asv.conf.json -o html || exit 1
python3.7 asv_exports.py || exit 1
