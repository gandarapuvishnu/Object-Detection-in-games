import os
from object_detection.object_detection_main import Detect

if __name__ == '__main__':

    PATH_TO_CKPT = 'object_detection/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('object_detection', 'data', 'mscoco_label_map.pbtxt')

    Detect(PATH_TO_CKPT, PATH_TO_LABELS )
