import tensorflow as tf
from google.protobuf import text_format
from object_detection import exporter
from object_detection.protos import pipeline_pb2
import config
import os

slim = tf.contrib.slim

def most_trained_checkpoint(path):
  most_trained_checkpoint_step_count = max(list(map(lambda x: int(x[len('model.ckpt-'):-len('.index')]), filter(lambda x: x.endswith('.index'), os.listdir(path)))))

  return os.path.join(path, 'model.ckpt-' + str(most_trained_checkpoint_step_count))

def go(input_type='image_tensor', pipeline_config_path=config.PIPELINE_CONFIG_PATH, output_directory=config.OUTPUT_INFERENCE_GRAPH_PATH, trained_checkpoint_prefix=None):

  if trained_checkpoint_prefix is None:
    trained_checkpoint_prefix = most_trained_checkpoint(config.TRAIN_PATH)

  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  with tf.gfile.GFile(pipeline_config_path, 'r') as f:
    text_format.Merge(f.read(), pipeline_config)

  exporter.export_inference_graph(
      input_type, pipeline_config, trained_checkpoint_prefix,
      output_directory)
