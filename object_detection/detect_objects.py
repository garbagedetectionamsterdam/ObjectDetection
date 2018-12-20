import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import json
import time
import glob

from io import StringIO
from PIL import Image

import matplotlib.pyplot as plt

from object_detection.utils import label_map_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

from multiprocessing.dummy import Pool as ThreadPool
import os
import config


MAX_NUMBER_OF_BOXES = 10
MINIMUM_CONFIDENCE = 0.9


pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.gfile.GFile(config.PIPELINE_CONFIG_PATH, 'r') as f:
    text_format.Merge(f.read(), pipeline_config)

# PATH_TO_LABELS = os.path.join(config.ANNOTATION_PATH, 'label_map.pbtxt')

label_map = label_map_util.load_labelmap(pipeline_config.train_input_reader.label_map_path)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=sys.maxsize, use_display_name=True)
CATEGORY_INDEX = label_map_util.create_category_index(categories)



class PredictionServer:

    def __init__(self):
        self.detection_graph = None
        self.last_load = None
        self.load_graph()

    def load_graph(self):
        print('load graph')
        graph_file_path = os.path.join(config.OUTPUT_INFERENCE_GRAPH_PATH, 'frozen_inference_graph.pb')

        if self.last_load is not None and os.path.isfile(graph_file_path) and self.last_load == os.path.getctime(graph_file_path):
            print('graph is already up to date')

            return
        print('update required, reloading graph')

        self.last_load = os.path.getctime(graph_file_path)

        # Load model into memory
        print('Loading model...')
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def detection_xml_string(self, boxes, class_names, image_width, image_height):
        result = """
            <annotation>
                <folder>less_selected</folder>
                <filename>undefined</filename>
                <size>
                    <width>""" + str(image_width) + """</width>
                    <height>""" + str(image_height) + """</height>
                </size>
                <segmented>0</segmented>
                """
        for i in range(len(class_names)):

            ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
            print(json.dumps(class_names[i]))
            print(json.dumps(ymin))

            print(json.dumps(ymax))

            result += """<object>
                    <name>""" + class_names[i]["name"] + """</name>
                    <pose>Unspecified</pose>
                    <truncated>0</truncated>
                    <difficult>0</difficult>
                    <bndbox>
                        <xmin>""" + str(xmin * image_width) + """</xmin>
                        <ymin>""" + str(ymin * image_height) + """</ymin>
                        <xmax>""" + str(xmax * image_width) + """</xmax>
                        <ymax>""" + str(ymax * image_height) + """</ymax>
                    </bndbox>
                </object>"""

        result += "</annotation>"

        return result

    def detect_objects(self, image_path):

        self.load_graph()
        print('detecting...')
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

                image = Image.open(image_path)
                image_np = self.load_image_into_numpy_array(image)
                image_np_expanded = np.expand_dims(image_np, axis=0)

                (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_np_expanded})

                box_filter = scores > MINIMUM_CONFIDENCE

                boxes = boxes[box_filter]
                classes = classes[box_filter]
                scores = scores[box_filter]

                class_names = []
                for i in range(classes.shape[0]):
                    class_names.append(CATEGORY_INDEX[classes[i]])

                return self.detection_xml_string(boxes, class_names, image.size[0], image.size[1])

