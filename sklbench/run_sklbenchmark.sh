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
asv run -b KMeans --append-samples || exit 1
asv run -b LogisticRegression --append-samples || exit 1
asv run -b LinearRegression --append-samples || exit 1
# asv run -b SGDRegressor --append-samples || exit 1
# asv run -b Ridge --append-samples || exit 1
# asv run -b ElasticNet --append-samples || exit 1
# asv run -b Lasso --append-samples || exit 1
# asv run -b PCA --append-samples || exit 1
# asv run -b SVC --append-samples || exit 1
# asv run -b KNeighbors --append-samples || exit 1

echo --PUBLISH-BENCHMARK--
if [ -d scikit-learn_benchmarks ]
then
    rm ./../dist/html/sklbench_results -r -f
fi
asv publish -o ./../dist/html/sklbench_results || exit 1

echo --END--
cd ..

