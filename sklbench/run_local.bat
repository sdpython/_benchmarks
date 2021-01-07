@echo off
set CDIR="%~dp0\.."

@echo --MODULES--
python -u %CDIR%/modules/bench_plot_re2.py --quiet
copy bench_re2*.csv %CDIR%\modules\results

@echo --SCIKIT-LEARN--
python -u %CDIR%/scikit-learn/bench_plot_gridsearch_cache.py --quiet
copy bench_plot_gridsearch_cache*.csv %CDIR%\scikit-learn\results
copy bench_plot_gridsearch_cache*.xlsx %CDIR%\scikit-learn\results

@echo --ONNXRUNTIME--
python -u %CDIR%/onnxruntime/test1-rf/bench_onnxruntime_rf_data.py
python -u %CDIR%/onnxruntime/test1-rf/bench_onnxruntime_rf.py
copy result*.csv %CDIR%/onnxruntime/test1-rf/results
copy result*.xlsx %CDIR%/onnxruntime/test1-rf/results

@echo --SKL2ONNX--
python %CDIR%/skl2onnx/bench_plot_onnxruntime_decision_tree.py --quiet
copy *.csv %CDIR%\skl2onnx\results
copy *.xlsx %CDIR%\skl2onnx\results

python %CDIR%/skl2onnx/bench_plot_onnxruntime_linreg.py --quiet
copy *.csv %CDIR%\skl2onnx\results
copy *.xlsx %CDIR%\skl2onnx\results

python %CDIR%/skl2onnx/bench_plot_onnxruntime_logreg.py --quiet
copy *.csv %CDIR%\skl2onnx\results
copy *.xlsx %CDIR%\skl2onnx\results

python %CDIR%/skl2onnx/bench_plot_onnxruntime_random_forest.py --quiet
copy *.csv %CDIR%\skl2onnx\results
copy *.xlsx %CDIR%\skl2onnx\results

python %CDIR%/skl2onnx/bench_plot_onnxruntime_random_forest_reg.py --quiet
copy *.csv %CDIR%\skl2onnx\results
copy *.xlsx %CDIR%\skl2onnx\results

python %CDIR%/skl2onnx/bench_plot_onnxruntime_svm_reg.py --quiet
copy *.csv %CDIR%\skl2onnx\results
copy *.xlsx %CDIR%\skl2onnx\results

python %CDIR%/skl2onnx/results --quiet
copy *.csv %CDIR%\skl2onnx\results
copy *.xlsx %CDIR%\skl2onnx\results

python %CDIR%/skl2onnx/run.sh --quiet
copy *.csv %CDIR%\skl2onnx\results
copy *.xlsx %CDIR%\skl2onnx\results

@echo --ONNX--
python %CDIR%/onnx/bench_plot_datasets_num.py --quiet
copy *.csv %CDIR%\onnx\results
copy *.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/bench_plot_datasets_num_reg.py --quiet
copy *.csv %CDIR%\onnx\results
copy *.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/bench_plot_datasets_num_reg_knn.py --quiet
copy *.csv %CDIR%\onnx\results
copy *.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/bench_plot_ml_ensemble.py --quiet
copy *.csv %CDIR%\onnx\results
copy *.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/bench_plot_onnxruntime_casc_add.py --quiet
copy *.csv %CDIR%\onnx\results
copy *.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/bench_plot_onnxruntime_casc_mlp.py --quiet
copy *.csv %CDIR%\onnx\results
copy *.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/bench_plot_onnxruntime_casc_scaler.py --quiet
copy *.csv %CDIR%\onnx\results
copy *.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/bench_plot_onnxruntime_decision_tree.py --quiet
copy *.csv %CDIR%\onnx\results
copy *.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/bench_plot_onnxruntime_decision_tree_reg.py --quiet
copy *.csv %CDIR%\onnx\results
copy *.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/bench_plot_onnxruntime_gbr.py --quiet
copy *.csv %CDIR%\onnx\results
copy *.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/bench_plot_onnxruntime_gpr.py --quiet
copy *.csv %CDIR%\onnx\results
copy *.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/bench_plot_onnxruntime_hgb.py --quiet
copy *.csv %CDIR%\onnx\results
copy *.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/bench_plot_onnxruntime_knn.py --quiet
copy *.csv %CDIR%\onnx\results
copy *.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/bench_plot_onnxruntime_logreg.py --quiet
copy *.csv %CDIR%\onnx\results
copy *.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/bench_plot_onnxruntime_mlp.py --quiet
copy *.csv %CDIR%\onnx\results
copy *.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/bench_plot_onnxruntime_random_forest.py --quiet
copy *.csv %CDIR%\onnx\results
copy *.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/bench_plot_onnxruntime_reduce_sum.node.0.png --quiet
copy *.csv %CDIR%\onnx\results
copy *.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/bench_plot_onnxruntime_reduce_sum.node.png --quiet
copy *.csv %CDIR%\onnx\results
copy *.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/bench_plot_onnxruntime_reduce_sum.perf.0.csv --quiet
copy *.csv %CDIR%\onnx\results
copy *.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/bench_plot_onnxruntime_reduce_sum.perf.csv --quiet
copy *.csv %CDIR%\onnx\results
copy *.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/bench_plot_onnxruntime_reduce_sum.py --quiet
copy *.csv %CDIR%\onnx\results
copy *.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/bench_plot_onnxruntime_reduce_sum.time.csv --quiet
copy *.csv %CDIR%\onnx\results
copy *.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/bench_plot_skl2onnx_unittest.py --quiet
copy *.csv %CDIR%\onnx\results
copy *.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/profiles --quiet
copy *.csv %CDIR%\onnx\results
copy *.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/profiles_reg --quiet
copy *.csv %CDIR%\onnx\results
copy *.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/profile_hgb.bat --quiet
copy *.csv %CDIR%\onnx\results
copy *.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/profile_hgb.py --quiet
copy *.csv %CDIR%\onnx\results
copy *.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/profile_reduce_sum.bat --quiet
copy *.csv %CDIR%\onnx\results
copy *.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/results --quiet
copy *.csv %CDIR%\onnx\results
copy *.xlsx %CDIR%\onnx\results


@echo --DONE--
