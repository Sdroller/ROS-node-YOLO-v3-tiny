#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""

import colorsys
import os
from timeit import default_timer as timer
import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from keras.backend.tensorflow_backend import set_session
from PIL import Image, ImageFont, ImageDraw
import cv2

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
import rospkg
from sdroller_track_detect.msg import bbox_array, bbox
import rospy

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#from keras.utils import multi_gpu_model
gpu_num = 1


class YOLO(object):
    def __init__(self, model, anchors, classes):
        self.model_path = model
        self.anchors_path = anchors
        self.classes_path = classes
        self.score = 0.3
        self.iou = 0.45
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()

        '''Control amount of memory used by YOLO. TX2 has shared memory between CPU and GPU.
        By default, TensorFlow maps nearly all of the GPU memory of all GPUs. It is desirable for the process to only
        allocate a subset of the available memory, or to only grow the memory usage as is needed by the process.
        TensorFlow provides two Config options on the Session to control this:

          - gpu_options.allow_growth: attempts to allocate only as much GPU memory based on runtime allocations.
                                      Note that there is a known error with this method as of Jetpack 3.3 (CUDA 9):
                                      It still takes a large chunk of memory (~4GB) instead of a small amount as needed.

          - gpu_options.per_process_gpu_memory_fraction: the fraction of the overall amount of memory that each visible
                                                         GPU should be allocated.
        '''
        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.2

        self.sess = tf.Session(config=config)
        K.set_session(self.sess)
        # set_session(tf.Session(config=config))
        # self.sess = K.get_session()

        self.model_image_size = (288, 288) #(320, 320)  # fixed size or (None, None), hw
        self.boxes, self.scores, self.classes = self.generate()
        self.graph = tf.get_default_graph()


    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        rospy.loginfo('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        # if gpu_num>=2:
        #    self.yolo_model = multi_gpu_model(self.yolo_model, gpus=gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()
        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)

        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        with self.graph.as_default():
            out_boxes, out_scores, out_classes = self.sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={
                    self.yolo_model.input: image_data,
                    self.input_image_shape: [image.shape[0], image.shape[1]],
                    K.learning_phase(): 1
                })
        
        # Create an array of all bounding boxes to be published later
        msg_yolov3_arr = bbox_array()
        msg_yolov3_arr.header.stamp = rospy.Time.now()

        # Print info and Draw Rectangles around detected objects
        str_info = '\n  YOLO detected {} boxes within img:'.format(len(out_boxes))
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            top, left, bottom, right = box # Note in image, ymin is at the top of image
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.shape[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.shape[1], np.floor(right + 0.5).astype('int32'))

            # For printing details of each detected object later
            str_info = str_info + ('\n    class: {},\t confidence: {:.2f},\t bbox: [({}, {}), ({}, {})]'.format(
                                    predicted_class, score, left, top, right, bottom))

            # Add rectangles with label to detected objects in image for visualization
            text_origin = (left + 5, top + 12)
            cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
            label = '{}, {:.2f}'.format(predicted_class, score)
            cv2.putText(image, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

            # Fill the bbox array
            msg_yolov3 = bbox()
            msg_yolov3.Class = predicted_class
            msg_yolov3.prob = score
            msg_yolov3.xmin = left
            msg_yolov3.ymin = top
            msg_yolov3.xmax = right
            msg_yolov3.ymax = bottom
            msg_yolov3.depth = -1.0 # This indicates valid value hasn't been filled out
            msg_yolov3.aspect_ratio = -1.0  # This indicates valid value hasn't been filled out
            msg_yolov3.area_percentage = -1.0   # This indicates valid value hasn't been filled out
            msg_yolov3_arr.bboxes.append(msg_yolov3)

        end = timer()
        str_info = str_info + '\n    YOLO FPS: %0.2f' % (1 / (end - start))
        rospy.loginfo(str_info)

        return image, msg_yolov3_arr


    def close_session(self):
        self.sess.close()


def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            rospy.logerr('Could not open image file! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()
