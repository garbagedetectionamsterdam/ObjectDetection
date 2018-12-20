import config
import os
import shutil
from object_detection.create_tf_record_go import go as create_tf_record
from object_detection.train_go import go as train
from object_detection.export_inference_graph_go import go as export_inference_graph

if os.path.isdir(config.TRAIN_PATH):
	shutil.rmtree(config.TRAIN_PATH)

create_tf_record()
train()
export_inference_graph(output_directory='./temp_output')

shutil.rmtree(config.OUTPUT_INFERENCE_GRAPH_PATH)
shutil.move('./temp_output', config.OUTPUT_INFERENCE_GRAPH_PATH)


