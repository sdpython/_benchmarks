echo --TRAIN--
cd onnxruntime/test1-rf
python -u bench_onnxruntime_rf_data.py
cd ../..

echo --INSTALL--
python -m pip install onnxruntime==1.2.0

echo --BENCH1--
cd onnxruntime/test1-rf
python -u bench_onnxruntime_rf.py
cd ../..

echo --INSTALL--
python -m pip install onnxruntime==1.3.0

echo --BENCH1--
cd onnxruntime/test1-rf
python -u bench_onnxruntime_rf.py
cd ../..

