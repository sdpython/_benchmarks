echo --PIP--
pip install git+https://github.com/sdpython/asv.git@jenkins

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
echo --BENCH--
python3.7 -m asv run --show-stderr --config asv.conf.json
if [ -d html ]
then
    echo --REMOVE HTML--
    rm html -r -f
fi
echo --PUBLISH--
python3.7 -m asv publish --config asv.conf.json -o html || exit 1
python3.7 -m mlprodict asv2csv -f results -o "asv_<date>.csv" || exit 1
