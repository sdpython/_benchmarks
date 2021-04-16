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

echo ---- RUN bench_plot_onnxruntime_decision_tree.py
set NAME=results\bench_plot_onnxruntime_decision_tree.csv
if exists %NAME% goto next_bench_plot_onnxruntime_decision_tree:
python %CDIR%/skl2onnx/bench_plot_onnxruntime_decision_tree.py --quiet
copy results\%NAME%.csv %CDIR%\skl2onnx\results
copy results\%NAME%.xlsx %CDIR%\skl2onnx\results
:next_bench_plot_onnxruntime_decision_tree:

echo ---- RUN bench_plot_onnxruntime_linreg.py
set NAME=results\bench_plot_onnxruntime_linreg.csv
if exists %NAME% goto next_bench_plot_onnxruntime_linreg:
python %CDIR%/skl2onnx/bench_plot_onnxruntime_linreg.py --quiet
copy results\%NAME%.csv %CDIR%\skl2onnx\results
copy results\%NAME%.xlsx %CDIR%\skl2onnx\results
:next_bench_plot_onnxruntime_linreg:

echo ---- RUN bench_plot_onnxruntime_logreg.py
set NAME=results\bench_plot_onnxruntime_logreg.csv
if exists %NAME% goto next_bench_plot_onnxruntime_logreg:
python %CDIR%/skl2onnx/bench_plot_onnxruntime_logreg.py --quiet
copy results\%NAME%.csv %CDIR%\skl2onnx\results
copy results\%NAME%.xlsx %CDIR%\skl2onnx\results
:next_bench_plot_onnxruntime_logreg:

echo ---- RUN bench_plot_onnxruntime_random_forest.py
set NAME=results\bench_plot_onnxruntime_random_forest.csv
if exists %NAME% goto next_bench_plot_onnxruntime_random_forest:
python %CDIR%/skl2onnx/bench_plot_onnxruntime_random_forest.py --quiet
copy results\%NAME%.csv %CDIR%\skl2onnx\results
copy results\%NAME%.xlsx %CDIR%\skl2onnx\results
:next_bench_plot_onnxruntime_random_forest:

echo ---- RUN bench_plot_onnxruntime_random_forest_reg.py
set NAME=results\bench_plot_onnxruntime_random_forest_reg.csv
if exists %NAME% goto next_bench_plot_onnxruntime_random_forest_reg:
python %CDIR%/skl2onnx/bench_plot_onnxruntime_random_forest_reg.py --quiet
copy results\%NAME%.csv %CDIR%\skl2onnx\results
copy results\%NAME%.xlsx %CDIR%\skl2onnx\results
:next_bench_plot_onnxruntime_random_forest_reg:

echo ---- RUN bench_plot_onnxruntime_svm_reg.py
set NAME=results\bench_plot_onnxruntime_svm_reg.csv
if exists %NAME% goto next_bench_plot_onnxruntime_svm_reg:
python %CDIR%/skl2onnx/bench_plot_onnxruntime_svm_reg.py --quiet
copy results\%NAME%.csv %CDIR%\skl2onnx\results
copy results\%NAME%.xlsx %CDIR%\skl2onnx\results
:next_bench_plot_onnxruntime_svm_reg:

:onnx:
@echo --ONNX--

echo ---- RUN bench_plot_datasets_num.py
set NAME=results\bench_plot_datasets_num.csv
if exists %NAME% goto next_bench_plot_datasets_num:
python %CDIR%/onnx/bench_plot_datasets_num.py --quiet
copy results\%NAME%.csv %CDIR%\onnx\results
copy results\%NAME%.xlsx %CDIR%\onnx\results
:next_bench_plot_datasets_num:

echo ---- RUN bench_plot_datasets_num_reg.py
set NAME=results\bench_plot_datasets_num_reg.csv
if exists %NAME% goto next_bench_plot_datasets_num_reg:
python %CDIR%/onnx/bench_plot_datasets_num_reg.py --quiet
copy results\%NAME%.csv %CDIR%\onnx\results
copy results\%NAME%.xlsx %CDIR%\onnx\results
:next_bench_plot_datasets_num_reg:

echo ---- RUN bench_plot_datasets_num_reg_knn.py
set NAME=results\bench_plot_datasets_num_reg_knn.csv
if exists %NAME% goto next_bench_plot_datasets_num_reg_knn:
python %CDIR%/onnx/bench_plot_datasets_num_reg_knn.py --quiet
copy results\%NAME%.csv %CDIR%\onnx\results
copy results\%NAME%.xlsx %CDIR%\onnx\results
:next_bench_plot_datasets_num_reg_knn:

echo ---- RUN bench_plot_ml_ensemble.py
set NAME=results\bench_plot_ml_ensemble.csv
if exists %NAME% goto next_bench_plot_ml_ensemble:
python %CDIR%/onnx/bench_plot_ml_ensemble.py --quiet
copy results\%NAME%.csv %CDIR%\onnx\results
copy results\%NAME%.xlsx %CDIR%\onnx\results
:next_bench_plot_ml_ensemble:

echo ---- RUN bench_plot_onnxruntime_casc_add.py
set NAME=results\bench_plot_onnxruntime_casc_add.csv
if exists %NAME% goto next_bench_plot_onnxruntime_casc_add:
python %CDIR%/onnx/bench_plot_onnxruntime_casc_add.py --quiet
copy results\%NAME%.csv %CDIR%\onnx\results
copy results\%NAME%.xlsx %CDIR%\onnx\results
:next_bench_plot_onnxruntime_casc_add:

echo ---- RUN bench_plot_onnxruntime_casc_mlp.py
set NAME=results\bench_plot_onnxruntime_casc_mlp.csv
if exists %NAME% goto next_bench_plot_onnxruntime_casc_mlp:
python %CDIR%/onnx/bench_plot_onnxruntime_casc_mlp.py --quiet
copy results\%NAME%.csv %CDIR%\onnx\results
copy results\%NAME%.xlsx %CDIR%\onnx\results
:next_bench_plot_onnxruntime_casc_mlp:

echo ---- RUN bench_plot_onnxruntime_casc_scaler.py
set NAME=results\bench_plot_onnxruntime_casc_scaler.csv
if exists %NAME% goto next_bench_plot_onnxruntime_casc_scaler:
python %CDIR%/onnx/bench_plot_onnxruntime_casc_scaler.py --quiet
copy results\%NAME%.csv %CDIR%\onnx\results
copy results\%NAME%.xlsx %CDIR%\onnx\results
:next_bench_plot_onnxruntime_casc_scaler:

echo ---- RUN bench_plot_onnxruntime_decision_tree.py
set NAME=results\bench_plot_onnxruntime_decision_tree.csv
if exists %NAME% goto next_bench_plot_onnxruntime_decision_tree:
python %CDIR%/onnx/bench_plot_onnxruntime_decision_tree.py --quiet
copy results\%NAME%.csv %CDIR%\onnx\results
copy results\%NAME%.xlsx %CDIR%\onnx\results
:next_bench_plot_onnxruntime_decision_tree:

echo ---- RUN bench_plot_onnxruntime_decision_tree_reg.py
set NAME=results\bench_plot_onnxruntime_decision_tree_reg.csv
if exists %NAME% goto next_bench_plot_onnxruntime_decision_tree_reg:
python %CDIR%/onnx/bench_plot_onnxruntime_decision_tree_reg.py --quiet
copy results\%NAME%.csv %CDIR%\onnx\results
copy results\%NAME%.xlsx %CDIR%\onnx\results
:next_bench_plot_onnxruntime_decision_tree_reg:

echo ---- RUN bench_plot_onnxruntime_gbr.py
set NAME=results\bench_plot_onnxruntime_gbr.csv
if exists %NAME% goto next_bench_plot_onnxruntime_gbr:
python %CDIR%/onnx/bench_plot_onnxruntime_gbr.py --quiet
copy results\%NAME%.csv %CDIR%\onnx\results
copy results\%NAME%.xlsx %CDIR%\onnx\results
:next_bench_plot_onnxruntime_gbr:

echo ---- RUN bench_plot_onnxruntime_gpr.py
set NAME=results\bench_plot_onnxruntime_gpr.csv
if exists %NAME% goto next_bench_plot_onnxruntime_gpr:
python %CDIR%/onnx/bench_plot_onnxruntime_gpr.py --quiet
copy results\%NAME%.csv %CDIR%\onnx\results
copy results\%NAME%.xlsx %CDIR%\onnx\results
:next_bench_plot_onnxruntime_gpr:

echo ---- RUN bench_plot_onnxruntime_hgb.py
set NAME=results\bench_plot_onnxruntime_hgb.csv
if exists %NAME% goto next_bench_plot_onnxruntime_hgb:
python %CDIR%/onnx/bench_plot_onnxruntime_hgb.py --quiet
copy results\%NAME%.csv %CDIR%\onnx\results
copy results\%NAME%.xlsx %CDIR%\onnx\results
:next_bench_plot_onnxruntime_hgb:

echo ---- RUN bench_plot_onnxruntime_knn.py
set NAME=results\bench_plot_onnxruntime_knn.csv
if exists %NAME% goto next_bench_plot_onnxruntime_knn:
python %CDIR%/onnx/bench_plot_onnxruntime_knn.py --quiet
copy results\%NAME%.csv %CDIR%\onnx\results
copy results\%NAME%.xlsx %CDIR%\onnx\results
:next_bench_plot_onnxruntime_knn:

echo ---- RUN bench_plot_onnxruntime_logreg.py
set NAME=results\bench_plot_onnxruntime_logreg.csv
if exists %NAME% goto next_bench_plot_onnxruntime_logreg:
python %CDIR%/onnx/bench_plot_onnxruntime_logreg.py --quiet
copy results\%NAME%.csv %CDIR%\onnx\results
copy results\%NAME%.xlsx %CDIR%\onnx\results
:next_bench_plot_onnxruntime_logreg:

echo ---- RUN bench_plot_onnxruntime_mlp.py
set NAME=results\bench_plot_onnxruntime_mlp.csv
if exists %NAME% goto next_bench_plot_onnxruntime_mlp:
python %CDIR%/onnx/bench_plot_onnxruntime_mlp.py --quiet
copy results\%NAME%.csv %CDIR%\onnx\results
copy results\%NAME%.xlsx %CDIR%\onnx\results
:next_bench_plot_onnxruntime_mlp:

echo ---- RUN bench_plot_onnxruntime_random_forest.py
set NAME=results\bench_plot_onnxruntime_random_forest.csv
if exists %NAME% goto next_bench_plot_onnxruntime_random_forest:
python %CDIR%/onnx/bench_plot_onnxruntime_random_forest.py --quiet
copy results\%NAME%.csv %CDIR%\onnx\results
copy results\%NAME%.xlsx %CDIR%\onnx\results
:next_bench_plot_onnxruntime_random_forest:

echo ---- RUN bench_plot_onnxruntime_reduce_sum.py
set NAME=results\bench_plot_onnxruntime_reduce_sum.csv
if exists %NAME% goto next_bench_plot_onnxruntime_reduce_sum:
python %CDIR%/onnx/bench_plot_onnxruntime_reduce_sum.py --quiet
copy results\%NAME%.csv %CDIR%\onnx\results
copy results\%NAME%.xlsx %CDIR%\onnx\results
:next_bench_plot_onnxruntime_reduce_sum:

echo ---- RUN bench_plot_skl2onnx_unittest.py
set NAME=results\bench_plot_skl2onnx_unittest.csv
if exists %NAME% goto next_bench_plot_skl2onnx_unittest:
python %CDIR%/onnx/bench_plot_skl2onnx_unittest.py --quiet
copy results\%NAME%.csv %CDIR%\onnx\results
copy results\%NAME%.xlsx %CDIR%\onnx\results
:next_bench_plot_skl2onnx_unittest:

echo ---- RUN profile_hgb.py
set NAME=results\profile_hgb.csv
if exists %NAME% goto next_profile_hgb:
python %CDIR%/onnx/profile_hgb.py --quiet
copy results\%NAME%.csv %CDIR%\onnx\results
copy results\%NAME%.xlsx %CDIR%\onnx\results
:next_profile_hgb:

@echo --DONE--