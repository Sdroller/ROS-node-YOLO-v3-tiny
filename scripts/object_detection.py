#!/usr/bin/env python

import time
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


class ObjectDetection:
    yolo = None
    config = None
    bridge = None
    topic_camera_input = '/123fcv_camera/image_raw'
    topic_yolo_output_img = '123yolo_output_img'

    def __init__(self):

        rospy.init_node('object_detection_node')

        rospack = rospkg.RosPack()
        path = rospack.get_path('object_detection')

        # Load config
        object_detection_config = rospy.get_param("/object_detection_config")
        self.config = yaml.load(open(object_detection_config))

        self.yolo = YOLO(path + self.config['classification']['model'], path + self.config['classification']['anchors'], path + self.config['classification']['classes'])
        self.bridge = CvBridge()

        # Subscribers
        self.topic_camera_input = self.config['topics']['topic_camera_input']
        self.topic_yolo_output_img = self.config['topics']['topic_yolo_output_img']
        rospy.Subscriber(self.topic_camera_input, Image, self.classify)

        # Publish the image with detection boxes drawn
        self.img_pub = rospy.Publisher(self.topic_yolo_output_img, Image, queue_size=10)

        rospy.spin()

    def classify(self, image):
        img = self.bridge.imgmsg_to_cv2(image, "bgr8")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes, scores, classes, img_ret = self.yolo.detect_image(img)

        # Publish output Img
        img_ret_ros = self.bridge.cv2_to_imgmsg(img_ret, encoding="rgb8")
        self.img_pub.publish(img_ret_ros)


if __name__ == '__main__':
    try:
        ObjectDetection()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start object detection node.')
