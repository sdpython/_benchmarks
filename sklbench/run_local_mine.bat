set PATH=C:\xadupre\venv\bench\Scripts;%PATH%
set PYTHONPATH=C:\xadupre\microsoft_xadupre\sklearn-onnx;C:\xadupre\microsoft_xadupre\onnxconverter-common;C:\xadupre\microsoft_xadupre\onnxmltools

@echo off
set CDIR="%~dp0\.."

@echo --MODULES--
python -u %CDIR%/modules/bench_plot_re2.py --quiet
copy bench_re2*.csv %CDIR%\modules\results

goto ort:
@echo --SCIKIT-LEARN--
python -u %CDIR%/scikit-learn/bench_plot_gridsearch_cache.py --quiet
copy bench_plot_gridsearch_cache*.csv %CDIR%\scikit-learn\results
copy bench_plot_gridsearch_cache*.xlsx %CDIR%\scikit-learn\results

:ort:
goto skl2onnx:
@echo --ONNXRUNTIME--
python -u %CDIR%/onnxruntime/test1-rf/bench_onnxruntime_rf_data.py
python -u %CDIR%/onnxruntime/test1-rf/bench_onnxruntime_rf.py
copy result*.csv %CDIR%/onnxruntime/test1-rf/results
copy result*.xlsx %CDIR%/onnxruntime/test1-rf/results

:skl2onnx:
@echo --SKL2ONNX--

python %CDIR%/skl2onnx/bench_plot_onnxruntime_decision_tree.py --quiet
copy results\bench_plot_onnxruntime_decision_tree*.csv %CDIR%\skl2onnx\results
copy results\bench_plot_onnxruntime_decision_tree*.xlsx %CDIR%\skl2onnx\results

python %CDIR%/skl2onnx/bench_plot_onnxruntime_linreg.py --quiet
copy results\bench_plot_onnxruntime_linreg*.csv %CDIR%\skl2onnx\results
copy results\bench_plot_onnxruntime_linreg*.xlsx %CDIR%\skl2onnx\results

python %CDIR%/skl2onnx/bench_plot_onnxruntime_logreg.py --quiet
copy results\bench_plot_onnxruntime_logreg*.csv %CDIR%\skl2onnx\results
copy results\bench_plot_onnxruntime_logreg*.xlsx %CDIR%\skl2onnx\results

python %CDIR%/skl2onnx/bench_plot_onnxruntime_random_forest.py --quiet
copy results\bench_plot_onnxruntime_random_forest*.csv %CDIR%\skl2onnx\results
copy results\bench_plot_onnxruntime_random_forest*.xlsx %CDIR%\skl2onnx\results

python %CDIR%/skl2onnx/bench_plot_onnxruntime_random_forest_reg.py --quiet
copy results\bench_plot_onnxruntime_random_forest_reg*.csv %CDIR%\skl2onnx\results
copy results\bench_plot_onnxruntime_random_forest_reg*.xlsx %CDIR%\skl2onnx\results

python %CDIR%/skl2onnx/bench_plot_onnxruntime_svm_reg.py --quiet
copy results\bench_plot_onnxruntime_svm_reg*.csv %CDIR%\skl2onnx\results
copy results\bench_plot_onnxruntime_svm_reg*.xlsx %CDIR%\skl2onnx\results

:onnx:
@echo --ONNX--

python %CDIR%/onnx/bench_plot_datasets_num.py --quiet
copy results\bench_plot_datasets_num*.csv %CDIR%\onnx\results
copy results\bench_plot_datasets_num*.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/bench_plot_datasets_num_reg.py --quiet
copy results\bench_plot_datasets_num_reg*.csv %CDIR%\onnx\results
copy results\bench_plot_datasets_num_reg*.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/bench_plot_datasets_num_reg_knn.py --quiet
copy results\bench_plot_datasets_num_reg_knn*.csv %CDIR%\onnx\results
copy results\bench_plot_datasets_num_reg_knn*.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/bench_plot_ml_ensemble.py --quiet
copy results\bench_plot_ml_ensemble*.csv %CDIR%\onnx\results
copy results\bench_plot_ml_ensemble*.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/bench_plot_onnxruntime_casc_add.py --quiet
copy results\bench_plot_onnxruntime_casc_add*.csv %CDIR%\onnx\results
copy results\bench_plot_onnxruntime_casc_add*.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/bench_plot_onnxruntime_casc_mlp.py --quiet
copy results\bench_plot_onnxruntime_casc_mlp*.csv %CDIR%\onnx\results
copy results\bench_plot_onnxruntime_casc_mlp*.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/bench_plot_onnxruntime_casc_scaler.py --quiet
copy results\bench_plot_onnxruntime_casc_scaler*.csv %CDIR%\onnx\results
copy results\bench_plot_onnxruntime_casc_scaler*.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/bench_plot_onnxruntime_decision_tree.py --quiet
copy results\bench_plot_onnxruntime_decision_tree*.csv %CDIR%\onnx\results
copy results\bench_plot_onnxruntime_decision_tree*.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/bench_plot_onnxruntime_decision_tree_reg.py --quiet
copy results\bench_plot_onnxruntime_decision_tree_reg*.csv %CDIR%\onnx\results
copy results\bench_plot_onnxruntime_decision_tree_reg*.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/bench_plot_onnxruntime_gbr.py --quiet
copy results\bench_plot_onnxruntime_gbr*.csv %CDIR%\onnx\results
copy results\bench_plot_onnxruntime_gbr*.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/bench_plot_onnxruntime_gpr.py --quiet
copy results\bench_plot_onnxruntime_gpr*.csv %CDIR%\onnx\results
copy results\bench_plot_onnxruntime_gpr*.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/bench_plot_onnxruntime_hgb.py --quiet
copy results\bench_plot_onnxruntime_hgb*.csv %CDIR%\onnx\results
copy results\bench_plot_onnxruntime_hgb*.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/bench_plot_onnxruntime_knn.py --quiet
copy results\bench_plot_onnxruntime_knn*.csv %CDIR%\onnx\results
copy results\bench_plot_onnxruntime_knn*.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/bench_plot_onnxruntime_logreg.py --quiet
copy results\bench_plot_onnxruntime_logreg*.csv %CDIR%\onnx\results
copy results\bench_plot_onnxruntime_logreg*.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/bench_plot_onnxruntime_mlp.py --quiet
copy results\bench_plot_onnxruntime_mlp*.csv %CDIR%\onnx\results
copy results\bench_plot_onnxruntime_mlp*.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/bench_plot_onnxruntime_random_forest.py --quiet
copy results\bench_plot_onnxruntime_random_forest*.csv %CDIR%\onnx\results
copy results\bench_plot_onnxruntime_random_forest*.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/bench_plot_onnxruntime_reduce_sum.py --quiet
copy results\bench_plot_onnxruntime_reduce_sum*.csv %CDIR%\onnx\results
copy results\bench_plot_onnxruntime_reduce_sum*.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/bench_plot_skl2onnx_unittest.py --quiet
copy results\bench_plot_skl2onnx_unittest*.csv %CDIR%\onnx\results
copy results\bench_plot_skl2onnx_unittest*.xlsx %CDIR%\onnx\results

python %CDIR%/onnx/profile_hgb.py --quiet
copy results\profile_hgb*.csv %CDIR%\onnx\results
copy results\profile_hgb*.xlsx %CDIR%\onnx\results


@echo --DONE--
