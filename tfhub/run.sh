echo --INSTALL--
python -m pip install "numpy<1.20" "flatbuffers<2.0" --no-deps

echo --RUN--
cd tfhub
python -u tfhub_mobilenet_v3_small_075_224.py || exit 1
cd ..

