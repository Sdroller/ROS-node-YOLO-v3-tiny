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


class ObjectDetection:
    yolo = None
    config = None
    bridge = None
    topic_camera_input = '/123fcv_camera/image_raw'
    topic_yolo_output_img = '123yolo_output_img'

    def __init__(self):
        rospack = rospkg.RosPack()
        path = rospack.get_path('object_detection')

        # Load config
        object_detection_config = rospy.get_param("/object_detection_config")
        self.config = yaml.load(open(object_detection_config))

        self.yolo = YOLO(path + self.config['classification']['model'], path + self.config['classification']['anchors'], path + self.config['classification']['classes'])
        self.bridge = CvBridge()

        # Subscribers
        self.topic_camera_input = self.config['topics']['topic_camera_input']
	    # https://stackoverflow.com/questions/26415699/ros-subscriber-not-up-to-date/29160379#29160379
        rospy.Subscriber(self.topic_camera_input, Image, self.classify, queue_size=1, buff_size=16777216)

        # Publish the image with detection boxes drawn
        self.topic_yolo_output_img = self.config['topics']['topic_yolo_output_img']
	self.img_pub = rospy.Publisher(self.topic_yolo_output_img, Image, queue_size=1)


    def classify(self, image):
        img = self.bridge.imgmsg_to_cv2(image, "bgr8")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes, scores, classes, img_ret = self.yolo.detect_image(img)

        # Publish output Img
        img_ret_ros = self.bridge.cv2_to_imgmsg(img_ret, encoding="rgb8")
        self.img_pub.publish(img_ret_ros)


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
