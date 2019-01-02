#!/usr/bin/env python

import time
import threading
import sys
import rospy
import message_filters
from std_msgs.msg import String
from std_msgs.msg import *
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import cv2
import numpy as np
import os
import csv
from classifier.yolo import YOLO
import rospkg
import yaml
from cv_bridge import CvBridge
from sdroller_track_detect.msg import bbox_array, bbox
from imutils.video import FPS
import copy


class ObjectDetectionTracking():
    def __init__(self):
        # Sub topics
        self.topic_rgb_image = '/zed/rgb/image_rect_color'
        self.topic_depth_image = '/zed/depth/depth_registered'  # Image: 32-bit depth values in meters
        # Pub topics
        self.topic_yolo_output_img = 'yolo/labelled_img'
        self.topic_tracker_output_img = 'labelled_img'

        self.new_img_received = False
        self.rgb_img = None
        self.rgb_img_output = None
        self.depth_img = None
        self.bridge = CvBridge()

        # Vars Tracker
        self.selected_target_bbox = None # Stores the bbox of the selected target, shortlisted from yolo_bbox_arr
        self.yolo_bbox_arr = [] # Stores array of bboxes returned by object detection code
        self.target_class = 'person'  # The class to be tracked.
        self.tracker_min_iou = 0.3  # Min value of IoU for a bbox to be considered same as tracked object.
        self.tracker_init_min_iou = 0.2  # This is used for first initialization of tracker only
        self.threshold_confidence = 0.4  # prob of class must be greater than this threshold for us to use it.
        self.tracker = None  # Object to store tracker: cv2.TrackerKCF_create() or cv2.TrackerMOSSE_create()
        self.fps = FPS().start()

        # Load yolo config
        rospack = rospkg.RosPack()
        path = rospack.get_path('sdroller_track_detect')
        object_detection_config = rospy.get_param("/object_detection_config")
        self.config = yaml.load(open(object_detection_config))
        self.yolo = YOLO(path + self.config['classification']['model'], path +
                         self.config['classification']['anchors'], path + self.config['classification']['classes'])

        # Subscribers
        '''We increase the default buff_size because it is smaller than the camera's images, and so if the node doesn't
        processing them fast enough, there will be a number of images backed up in some queue and it appears as a
        lag in the video stream.
        We also use the TimeSynchronizer to subscribe to multiple topics.
        '''
        buff_size_bytes = 8 * 1024 * 1024  # 8MB
        self.rgb_img_sub = message_filters.Subscriber(
            self.topic_rgb_image, Image, queue_size=1, buff_size=buff_size_bytes)
        self.depth_img_sub = message_filters.Subscriber(
            self.topic_depth_image, Image, queue_size=1, buff_size=buff_size_bytes)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_img_sub, self.depth_img_sub], 1, 0.1, allow_headerless=False)
        self.ts.registerCallback(self.callback_rgb_depth_imgs)

        # Publish the image with detection boxes drawn
        self.yolo_labelled_img_pub = rospy.Publisher(self.topic_yolo_output_img, Image, queue_size=1)
        self.tracker_labelled_img_pub = rospy.Publisher(self.topic_tracker_output_img, Image, queue_size=1)

        # Get image details
        rgb_img_sample = rospy.wait_for_message(self.topic_rgb_image, Image)
        self.img_height = rgb_img_sample.height
        self.img_width = rgb_img_sample.width

        # Set the value of tracker_bbox to middle of the image, where a person normally detected standing in front of robot
        self.tracker_bbox = bbox()
        self.tracker_bbox.xmin = int(self.img_width * (1.0 / 3))
        self.tracker_bbox.ymin = int(self.img_height * (1.0 / 4))
        self.tracker_bbox.xmax = int(self.img_width * (2.0 / 3))
        self.tracker_bbox.ymax = int(self.img_height)

    def callback_rgb_depth_imgs(self, rgb_img, depth_img):
        # Get the depth img
        if(depth_img.encoding == '32FC1'):
            try:
                self.depth_img = self.bridge.imgmsg_to_cv2(depth_img, desired_encoding="32FC1")
            except CvBridgeError as e:
                rospy.logerr('%s' % (e))
                return
        else:
            rospy.logerr("ERROR: Depth Images from Stereo camera are not in 32FC1 format")
            return

        # Get the RGB img for tracker
        try:
            self.rgb_img = self.bridge.imgmsg_to_cv2(rgb_img, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr('%s' % (e))
            return

        self.new_img_received = True

        # Check for valid data
        if self.rgb_img is None:
            rospy.logwarn('RGB img is None!')
            self.new_img_received = False
        if self.depth_img is None:
            rospy.logwarn('Depth img is None!')
            self.new_img_received = False

    def classify(self, bgr_img):
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        labelled_rgb_img, bbox_arr_detections = self.yolo.detect_image(rgb_img)

        # Only store bboxes belonging to certain class
        self.select_target_class_bboxes(bbox_arr_detections)

        # Publish img labelled with all bboxes detected by yolo
        labelled_img_ros = self.bridge.cv2_to_imgmsg(labelled_rgb_img, encoding="rgb8")
        self.yolo_labelled_img_pub.publish(labelled_img_ros)

    def calc_iou(self, boxA, boxB):
        '''
        Calculates either overlap or Intersection over Union for 2 bounding boxes

        Args:
            base: If none, Intersection over Union is calculated.
                Else, boxA treated as base bounding box, over which overlap of the other box will be calculated.
            boxA (bbox): The target bounding box from object detection
            boxB (bbox): The bounding box from the tracker. This is the target bbox.

        Return:
            float: IoU as a percentage.
        '''

        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA.xmin, boxB.xmin)
        yA = max(boxA.ymin, boxB.ymin)
        xB = min(boxA.xmax, boxB.xmax)
        yB = min(boxA.ymax, boxB.ymax)

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA.xmax - boxA.xmin + 1) * (boxA.ymax - boxA.ymin + 1)
        boxBArea = (boxB.xmax - boxB.xmin + 1) * (boxB.ymax - boxB.ymin + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea)

        # return the intersection over union value
        return iou

    def select_viable_bbox(self, min_iou):
        '''This function selects a bbox from the array of yolo bboxes that is the most viable target.
        A viable target has a minimum IoU with the tracker's bbox. The most viable target has the highest IoU

        Args:
            None

        Returns:
            yolo_bbox (bbox): None if no viable target found, else most viable bbox
        '''

        selected_target_bbox = None
        prev_iou = 0
        for bbox in self.yolo_bbox_arr:  # yolo_bbox_arr contains only bbox of target class which we want to follow
            iou = max(self.calc_iou(bbox, self.tracker_bbox), self.calc_iou(self.tracker_bbox, bbox))
            if ((iou > min_iou) and (iou > prev_iou)):
                prev_iou = iou
                selected_target_bbox = bbox

        return selected_target_bbox

    def select_target_class_bboxes(self, bbox_array):
        '''This function selects all the bboxes that are of the target class and have greater confidence than a
        threshold. These bboxes are added to the self.yolo_bbox_arr list.

        Args:
            bbox_array (bbox[]): An array of bboxes

        Returns:
            None
        '''
        self.yolo_bbox_arr = []
        for bbox in bbox_array.bboxes:
            if (bbox.Class == self.target_class) and (bbox.prob > self.threshold_confidence):
                xmin, xmax = min(bbox.xmin, bbox.xmax), max(bbox.xmin, bbox.xmax)
                ymin, ymax = min(bbox.ymin, bbox.ymax), max(bbox.ymin, bbox.ymax)
                width = (xmax - xmin)
                height = (ymax - ymin)
                # Modify the bounding box from YOLO, for example to cut off the lower half
                bbox.xmin = int(xmin + width/4)
                bbox.xmax = int(xmax - width/4)
                bbox.ymin = int(ymin + height/4)
                bbox.ymax = int(ymax + 0) #)
                self.yolo_bbox_arr.append(bbox)

    def initialize_tracker(self, input_img):
        x, y = self.selected_target_bbox.xmin, self.selected_target_bbox.ymin
        w = self.selected_target_bbox.xmax - self.selected_target_bbox.xmin
        h = self.selected_target_bbox.ymax - self.selected_target_bbox.ymin
        bbox = (x, y, w, h)

        self.tracker = cv2.TrackerKCF_create() # cv2.TrackerMOSSE_create()
        ok = self.tracker.init(input_img, bbox)

        return ok

    def tracker_update(self, input_img):
        '''The tracker's bbox will be centre of image at startup. Target must be inside this area to init tracker
        '''

        if self.tracker is None:
            # Draw the tracker_bbox and yolo_bbox_arr for visualization purposes.
            text_origin = (self.tracker_bbox.xmin + 10, self.tracker_bbox.ymin + 25)
            cv2.putText(self.rgb_img_output, 'Waiting to detect target...', text_origin,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(self.rgb_img_output, (self.tracker_bbox.xmin, self.tracker_bbox.ymin),
                          (self.tracker_bbox.xmax, self.tracker_bbox.ymax), (255, 0, 0), 2)
            for bbox in self.yolo_bbox_arr:
                cv2.rectangle(self.rgb_img_output, (bbox.xmin, bbox.ymin),
                              (bbox.xmax, bbox.ymax), (0, 0, 255), 2)

            # Publish labelled Img
            img_ret_ros = self.bridge.cv2_to_imgmsg(self.rgb_img_output, encoding="bgr8")
            self.tracker_labelled_img_pub.publish(img_ret_ros)

            # If target class detected, search for viable target that fits this criteria.
            if len(self.yolo_bbox_arr) > 0:
                # Choose the bbox with the highest IoU (intersection over union) with centre of image
                self.selected_target_bbox = self.select_viable_bbox(self.tracker_init_min_iou)
                self.yolo_bbox_arr = []
                if not self.selected_target_bbox is None:
                    rospy.loginfo('Initializing Tracker on New Target...')
                    success = self.initialize_tracker(input_img)
                    if success:
                        rospy.loginfo('Target Acquired!!!')
                    else:
                        self.tracker = None
                        rospy.logwarn('Could not initialize tracker from YOLO bbox')

        else:
            # If targets detected, search for viable target and re-init tracker
            if len(self.yolo_bbox_arr) > 0:
                # Draw all bboxes on output img for debugging
                for bbox in self.yolo_bbox_arr:
                    cv2.rectangle(self.rgb_img_output, (bbox.xmin, bbox.ymin),
                                  (bbox.xmax, bbox.ymax), (0, 0, 255), 6)

                # Choose the bbox with the highest IoU (intersection over union) with prev bbox of tracker
                self.selected_target_bbox = self.select_viable_bbox(self.tracker_min_iou)
                self.yolo_bbox_arr = []
                if not self.selected_target_bbox is None:
                    success = self.initialize_tracker(input_img)
                    if success:
                        rospy.loginfo('Tracker ReInitialized from YOLO bbox!')
                    if not success:
                        rospy.logwarn('Failure to re-initialize Tracker from object detections bbox!')

            (success, box) = self.tracker.update(input_img)
            if not success:
                rospy.logwarn('Tracker Update Failure!')

                text_origin = (int(self.img_width * 0.1), int(self.img_height * 0.15))
                cv2.putText(self.rgb_img_output, 'Tracking Failure!', text_origin,
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            else:
                (x, y, w, h) = [(v) for v in box]
                self.tracker_bbox.ymin = max(0, np.floor(y + 0.5).astype('int32'))
                self.tracker_bbox.xmin = max(0, np.floor(x + 0.5).astype('int32'))
                self.tracker_bbox.ymax = min(self.img_height, np.floor(y + h + 0.5).astype('int32'))
                self.tracker_bbox.xmax = min(self.img_width, np.floor(x + w + 0.5).astype('int32'))

                # Publish output Img
                text_origin = (self.tracker_bbox.xmin + 10, self.tracker_bbox.ymin + 25)
                cv2.putText(self.rgb_img_output, 'Target Acquired', text_origin,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(self.rgb_img_output, (self.tracker_bbox.xmin, self.tracker_bbox.ymin),
                              (self.tracker_bbox.xmax, self.tracker_bbox.ymax), (255, 0, 0), 2)


            img_ret_ros = self.bridge.cv2_to_imgmsg(self.rgb_img_output, encoding="bgr8")
            self.tracker_labelled_img_pub.publish(img_ret_ros)

            #update the FPS counter
            self.fps.update()
            self.fps.stop()
            rospy.loginfo("Tracking FPS: {:.2f}".format(self.fps.fps()) )


def main(args):
    rospy.init_node('sdroller_track_detect_node')

    yolo = ObjectDetectionTracking()

    start_time = time.time()
    counter = 0

    while (not rospy.is_shutdown()):
        if yolo.new_img_received:
            yolo.new_img_received = False
            input_img = copy.deepcopy(yolo.rgb_img)
            yolo.rgb_img_output = copy.deepcopy(yolo.rgb_img)

            # Run yolo every 1 sec
            if (time.time() - start_time > counter):
                counter += 1
                yolo.classify(input_img)

            yolo.tracker_update(input_img)


if __name__ == '__main__':
    main(sys.argv)
