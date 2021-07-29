echo --INSTALL--
python -m pip install tensorflow tensorflow_hub

echo --RUN--
cd tfhub
python tfhub_mobilenet_v3_small_075_224.py
cd ..

