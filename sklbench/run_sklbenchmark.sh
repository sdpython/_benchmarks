echo --CLONE--
git clone https://github.com/sdpython/scikit-learn_benchmarks
cd scikit-learn_benchmarks

echo --RUN-BENCHMARK--
# asv run -b _bench --python=./_venv/bin/python
asv run -b _bench

echo --PUBLISH-BENCHMARK--
mkdir dist
asv publish -o ./../dist/html

echo --END--
cd ..

