#!/usr/bin/env python

import time
import sys
import rospy
from std_msgs.msg import String
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
import numpy as np
import os
import csv
from classifier.yolo import YOLO
import rospkg
import yaml
from cv_bridge import CvBridge
from object_detection.msg import bbox_array, bbox

class ObjectDetection():
    yolo = None
    config = None
    bridge = None

    def __init__(self):
        self.topic_camera_input = '/camera/image_raw'
        self.topic_yolo_output_img = 'labelled_img'
        self.topic_yolo_bbox_array = 'bbox_arr'

        rospack = rospkg.RosPack()
        path = rospack.get_path('object_detection')

        # Load config
        object_detection_config = rospy.get_param("/object_detection_config")
        self.config = yaml.load(open(object_detection_config))

        self.yolo = YOLO(path + self.config['classification']['model'], path + self.config['classification']['anchors'], path + self.config['classification']['classes'])
        self.bridge = CvBridge()

        # Subscribers
        # https://stackoverflow.com/questions/26415699/ros-subscriber-not-up-to-date/29160379#29160379
        rospy.Subscriber(self.topic_camera_input, Image, self.classify, queue_size=1, buff_size=16777216)

        # Publish the image with detection boxes drawn
        self.img_pub = rospy.Publisher(self.topic_yolo_output_img, Image, queue_size=1)
        self.bbox_pub = rospy.Publisher(self.topic_yolo_bbox_array, bbox_array, queue_size=1)

    def classify(self, image):
        img = self.bridge.imgmsg_to_cv2(image, "bgr8")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes, scores, classes, img_ret = self.yolo.detect_image(img)

        # Publish output Img
        img_ret_ros = self.bridge.cv2_to_imgmsg(img_ret, encoding="rgb8")
        self.img_pub.publish(img_ret_ros)

        # Publish Bounding Boxes and Probabilities
        msg_yolov3_arr = bbox_array()
        for _box, _score, _class in zip(boxes, scores, classes):
            msg_yolov3 = bbox()
            msg_yolov3.Class = str(_class)
            msg_yolov3.prob = _score
            msg_yolov3.xmin = _box[0]
            msg_yolov3.ymin = _box[1]
            msg_yolov3.xmax = _box[2]
            msg_yolov3.ymax = _box[3]
            msg_yolov3_arr.bboxes.append(msg_yolov3)

        self.bbox_pub.publish(msg_yolov3_arr)



def main(args):
    rospy.init_node('object_detection_node')

    yolo = ObjectDetection()

    try:
        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main(sys.argv)
