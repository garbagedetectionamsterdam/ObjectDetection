export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
source env/bin/activate
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
FLASK_APP=api.py flask run --host=0.0.0.0
