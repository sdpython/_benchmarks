echo --INSTALL--
python -m pip install tensorflow tensorflow_hub --no-deps

echo --RUN--
cd tfhub
python -u tfhub_mobilenet_v3_small_075_224.py
cd ..

