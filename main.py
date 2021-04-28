import numpy as np
import cv2
import tensorflow as tf
from utils import label_map_util
from utils import ops as utils_ops
from utils import visualization_utils as vis_util
import os
from utils.grabscreen import grab_screen


def load_model():
    return tf.keras.models.load_model \
        (filepath=
         os.path.join('./inference/ssd_inception_v2_coco_2017_11_17/saved_model/'))


def detect(img):
    image = np.asarray(img)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # patch tf1 into `utils.ops`
    utils_ops.tf = tf.compat.v1
    # Patch the location of gfile
    tf.gfile = tf.io.gfile

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return image, output_dict


def showInMovedWindow(winname, img, x, y):
    cv2.namedWindow(winname)  # Create a named window
    cv2.moveWindow(winname, x, y)  # Move it to (x,y)
    cv2.imshow(winname, img)


if __name__ == '__main__':

    GAME_WIDTH, GAME_HEIGHT = 1600, 1200

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print('Allocation Error: {}'.format(str(e)))

    model = load_model()

    Width, Height = 640, 480

    label_map = label_map_util.load_labelmap('./inference/mscoco_label_map.pbtxt')
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    while True:
        frame = grab_screen(region=(0, 40, GAME_WIDTH, GAME_HEIGHT))
        image_np, output = detect(frame)

        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output['detection_boxes'],
            output['detection_classes'],
            output['detection_scores'],
            category_index,
            instance_masks=output.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=8)

        temp = cv2.cvtColor(cv2.resize(image_np, (Width, Height)),
                            cv2.COLOR_BGR2RGB)

        showInMovedWindow('Testing', temp, 1300, 200)

        cv2.waitKey(1)
