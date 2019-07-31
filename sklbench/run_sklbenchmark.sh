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
asv run -b KMeans || exit 1
asv run -b LogisticRegression || exit 1
asv run -b LinearRegression || exit 1
asv run -b SGDRegressor || exit 1
asv run -b Ridge || exit 1
asv run -b ElasticNet || exit 1
asv run -b Lasso || exit 1
asv run -b PCA || exit 1
asv run -b SVC || exit 1
asv run -b KNeighbors || exit 1

echo --PUBLISH-BENCHMARK--
if [ -d scikit-learn_benchmarks ]
then
    rm ./../dist/html/sklbench_results -r -f
fi
asv publish -o ./../dist/html/sklbench_results || exit 1

echo --END--
cd ..

