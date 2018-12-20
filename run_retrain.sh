
# source /home/ubuntu/.dlamirc
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

source env/bin/activate

python retrain.py
