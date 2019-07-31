echo --CLONE--
git clone https://github.com/sdpython/scikit-learn_benchmarks
cd scikit-learn_benchmarks

echo --MACHINE--
asv machine --yes || exit 1

echo --RUN-BENCHMARK--
# asv run -b _bench --python=./_venv/bin/python
asv run -b _bench || exit 1

echo --PUBLISH-BENCHMARK--
asv publish -o ./../dist/html || exit 1

echo --END--
cd ..

