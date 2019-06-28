#/bin/bash
export PYTHONPATH=./python
python examples/keras/simple/mlp.py --data ./examples/keras/simple/data/SansEC_Train_Data.csv --model examples/keras/simple/best_model.h5 --xclbin xclbins/$1/gemx.xclbin --cfg xclbins/$1/config_info.dat --gemxlib ./C++/lib/libgemxhost.so
