cd skl2onnx
echo --BENCHMARK1--
python -u bench_plot_onnxruntime_random_forest_reg.py || exit 1
echo --BENCHMARK2--
python -u bench_plot_onnxruntime_svm_reg.py || exit 1
echo --BENCHMARK3--
python -u bench_plot_onnxruntime_logreg.py || exit 1
echo --BENCHMARK4--
python -u bench_plot_onnxruntime_linreg.py || exit 1
echo --BENCHMARK-GRAPH--
cd results
python -u post_graph.py || exit 1
cd ../..