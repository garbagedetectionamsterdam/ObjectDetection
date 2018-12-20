import functools
import json
import os
import tensorflow as tf

from google.protobuf import text_format

from object_detection import trainer
from object_detection.builders import input_reader_builder
from object_detection.builders import model_builder
from object_detection.protos import input_reader_pb2
from object_detection.protos import model_pb2
from object_detection.protos import pipeline_pb2
from object_detection.protos import train_pb2
import config

tf.logging.set_verbosity(tf.logging.INFO)

def get_configs_from_pipeline_file(pipeline_config_path):
  """Reads training configuration from a pipeline_pb2.TrainEvalPipelineConfig.

  Reads training config from file specified by pipeline_config_path flag.

  Returns:
    model_config: model_pb2.DetectionModel
    train_config: train_pb2.TrainConfig
    input_config: input_reader_pb2.InputReader
  """
  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  with tf.gfile.GFile(pipeline_config_path, 'r') as f:
    text_format.Merge(f.read(), pipeline_config)

  model_config = pipeline_config.model
  train_config = pipeline_config.train_config
  input_config = pipeline_config.train_input_reader

  return model_config, train_config, input_config


def get_configs_from_multiple_files(train_config_path, model_config_path, input_config_path):
  """Reads training configuration from multiple config files.

  Reads the training config from the following files:
    model_config: Read from --model_config_path
    train_config: Read from --train_config_path
    input_config: Read from --input_config_path

  Returns:
    model_config: model_pb2.DetectionModel
    train_config: train_pb2.TrainConfig
    input_config: input_reader_pb2.InputReader
  """
  train_config = train_pb2.TrainConfig()
  with tf.gfile.GFile(train_config_path, 'r') as f:
    text_format.Merge(f.read(), train_config)

  model_config = model_pb2.DetectionModel()
  with tf.gfile.GFile(model_config_path, 'r') as f:
    text_format.Merge(f.read(), model_config)

  input_config = input_reader_pb2.InputReader()
  with tf.gfile.GFile(input_config_path, 'r') as f:
    text_format.Merge(f.read(), input_config)

  return model_config, train_config, input_config

def go(train_dir=config.TRAIN_PATH, pipeline_config_path=config.PIPELINE_CONFIG_PATH, num_clones=1, clone_on_cpu=False):

  if pipeline_config_path:
    model_config, train_config, input_config = get_configs_from_pipeline_file(pipeline_config_path)
  else:
    model_config, train_config, input_config = get_configs_from_multiple_files()

  model_fn = functools.partial(
      model_builder.build,
      model_config=model_config,
      is_training=True)

  create_input_dict_fn = functools.partial(
      input_reader_builder.build, input_config)

  env = json.loads(os.environ.get('TF_CONFIG', '{}'))
  cluster_data = env.get('cluster', None)
  cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None
  task_data = env.get('task', None) or {'type': 'master', 'index': 0}
  task_info = type('TaskSpec', (object,), task_data)

  # Parameters for a single worker.
  ps_tasks = 0
  worker_replicas = 1
  worker_job_name = 'lonely_worker'
  task = 0
  is_chief = True
  master = ''

  if cluster_data and 'worker' in cluster_data:
    # Number of total worker replicas include "worker"s and the "master".
    worker_replicas = len(cluster_data['worker']) + 1
  if cluster_data and 'ps' in cluster_data:
    ps_tasks = len(cluster_data['ps'])

  if worker_replicas > 1 and ps_tasks < 1:
    raise ValueError('At least 1 ps task is needed for distributed training.')

  if worker_replicas >= 1 and ps_tasks > 0:
    # Set up distributed training.
    server = tf.train.Server(tf.train.ClusterSpec(cluster), protocol='grpc',
                             job_name=task_info.type,
                             task_index=task_info.index)
    if task_info.type == 'ps':
      server.join()
      return

    worker_job_name = '%s/task:%d' % (task_info.type, task_info.index)
    task = task_info.index
    is_chief = (task_info.type == 'master')
    master = server.target

  trainer.train(create_input_dict_fn, model_fn, train_config, master, task,
                num_clones, worker_replicas, clone_on_cpu, ps_tasks,
                worker_job_name, is_chief, train_dir)
