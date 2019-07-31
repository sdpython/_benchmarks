echo --CLONE--
if [ ! -d scikit-learn_benchmarks ]
then
    git clone https://github.com/sdpython/scikit-learn_benchmarks --recursive
else
    cd scikit-learn_benchmarks
    git pull
    git submodule update --init --recursive
    cd ..
fi

cd scikit-learn_benchmarks

echo --MACHINE--
asv machine --yes || exit 1

echo --RUN-BENCHMARK--
# asv run -b _bench --python=./_venv/bin/python
asv run -b _bench || exit 1

echo --PUBLISH-BENCHMARK--
if [ -d scikit-learn_benchmarks ]
then
    rm ./../dist/html -r -f
fi
asv publish -o ./../dist/html || exit 1

echo --END--
cd ..

