#!/usr/bin/env python

import time
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
from classifier.yolo import YOLO
import rospkg
import yaml
from cv_bridge import CvBridge
from sdroller_track_detect.msg import bbox_array, bbox
from imutils.video import FPS
import math

class ObjectDetectionTracking():
    def __init__(self):
        # Load yolo config
        rospack = rospkg.RosPack()
        path = rospack.get_path('sdroller_track_detect')
        object_detection_config = rospy.get_param("/object_detection_config")
        self.config = yaml.load(open(object_detection_config))
        self.yolo = YOLO(path + self.config['classification']['model'], path +
                         self.config['classification']['anchors'], path + self.config['classification']['classes'])
        
        # Sub topics
        self.topic_rgb_image = self.config['topics']['topic_rgb_image']       #'/zed/rgb/image_rect_color'
        self.topic_depth_image = self.config['topics']['topic_depth_image']   #'/zed/depth/depth_registered'  # Image: 32-bit depth values in meters
        # Pub topics
        self.topic_yolo_output_img = self.config['topics']['topic_yolo_output']     #'yolo/labelled_img'
        self.topic_tracker_output_img = self.config['topics']['topic_tracker_output']  #'labelled_img'

        # Vars Callbacks
        self.new_img_received = False
        self.rgb_img = None
        self.depth_img = None
        self.bridge = CvBridge()

        # Vars Tracker
        self.tracker_bbox = None            # The bbox of the target to be tracked
        self.most_viable_target_bbox = None # Stores the bbox of the 1 selected target out of valid targets list
        self._valid_targets_bbox_arr  = []   # Stores array of bboxes that can be considered valid targets
        
        self.target_class         = self.config['tracker']['target_class']          # The class to be tracked.
        self.tracker_init_min_iou = self.config['tracker']['tracker_init_min_iou']  # IoU of target bbox with tracker bbox. This is used for first initialization of tracker only
        self.threshold_confidence = self.config['tracker']['threshold_confidence']  # Min confidence of a detection for it to be considered a valid target
        self.target_max_dist      = self.config['tracker']['target_max_dist']       # Max dist of target from camera in meters for it to be considered a valid target
        self.target_min_aspect_ratio    = self.config['tracker']['min_aspect_ratio']
        self.target_max_aspect_ratio    = self.config['tracker']['max_aspect_ratio']
        self.target_min_area_percentage = self.config['tracker']['min_area_percentage']
        self.tracker_lost_count         = 0 #The number of consecutive frames that a viable target is not found.
        self.tracker_max_lost_count     = self.config['tracker']['tracker_max_lost_count'] # Max number of frames that a viable target is not found before declaring that target is lost.
        self.target_lost = False            # Indicates if the target is lost
        self.tracker_first_target_found = False # Indicates if the tracking process has started. Used to prevent robot from calling target_lost on startup.
        self.tracker_update_successful = False

        self.fps = FPS().start()

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

        # Publishers : publishers the rgb image with detection boxes drawn on it
        self.yolo_labelled_img_pub = rospy.Publisher(self.topic_yolo_output_img, Image, queue_size=1)
        self.tracker_labelled_img_pub = rospy.Publisher(self.topic_tracker_output_img, Image, queue_size=1)

        # Get image details
        rgb_img_sample = rospy.wait_for_message(self.topic_rgb_image, Image)
        self.img_height = rgb_img_sample.height
        self.img_width = rgb_img_sample.width

        # Init Tracker
        self.tracker_init()
        

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
        '''Runs the object detection model on input image to generate bounding boxes of detected objects.
        Also publishes input image with the detected bounding boxes drawn on it.
        '''
        # Detect Objects within images
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        labelled_rgb_img, bbox_arr_detections = self.yolo.detect_image(rgb_img)

        # Publish raw yolo output: img labelled with all bboxes detected
        labelled_img_ros = self.bridge.cv2_to_imgmsg(labelled_rgb_img, encoding="rgb8")
        self.yolo_labelled_img_pub.publish(labelled_img_ros)
     
        return bbox_arr_detections
        

    def _filter_detected_bboxes(self, msg_bbox_array, depth_img):
        '''This function selects all the bboxes that meets required criteria.
        For a bbox to be a valid target, it must be of the target class, have a confidence greater than the threshold,
        be of a max depth away from camera, have min area and a min aspect ratio. 
        
        All the bboxes that meet the criteria are added to the self.valid_targets_bbox_arr list.

        Args:
            msg_bbox_array (Msg bbox[]): An array of bboxes
            depth_img (np.ndarray, float32 format): Depth image
        Returns:
            None
        '''
        self._valid_targets_bbox_arr = []
        for bbox in msg_bbox_array.bboxes:
            # Check is bbox belongs to target class and has confidence above threshold
            if (bbox.Class == self.target_class) and (bbox.prob > self.threshold_confidence):

                # Check aspect ratio and area of bbox
                bbox_width  = (bbox.xmax - bbox.xmin)
                bbox_height = (bbox.ymax - bbox.ymin)
                bbox_aspect_ratio    = float(bbox_width) / bbox_height
                bbox_area_percentage = float(bbox_width * bbox_height) / (self.img_width * self.img_height)
                

                if ((bbox_aspect_ratio > self.target_min_aspect_ratio) and 
                    (bbox_aspect_ratio < self.target_max_aspect_ratio) and 
                    (bbox_area_percentage > self.target_min_area_percentage)):
                    
                    # Calculate Centroid of bbox
                    centroid_x = (bbox.xmin + bbox.xmax) // 2
                    centroid_y = (bbox.ymin + bbox.ymax) // 2
                    centroid_bbox = [centroid_x, centroid_y]

                    # Take mean depth of target from NxN size window around centre of bbox
                    window_size = 10 // 2 #the width of window around centroid which we sample for depth
                    centroid_bbox[0] = np.clip(centroid_bbox[0], (0 + window_size), (self.img_width  - window_size) )
                    centroid_bbox[1] = np.clip(centroid_bbox[1], (0 + window_size), (self.img_height - window_size) )
                    mean_depth_image = depth_img[centroid_bbox[1]-window_size : centroid_bbox[1]+window_size,\
                                                centroid_bbox[0]-window_size : centroid_bbox[0]+window_size]

                    mean_depth_image[~np.isfinite(mean_depth_image)] = 0.0 # Clean the NaN and +/- inf values
                    distance_to_target = np.mean(mean_depth_image)
                    

                    # Check if bbox has distance less than max limit
                    if (bbox.depth < self.target_max_dist):
                       
                        bbox.depth = distance_to_target
                        bbox.aspect_ratio = bbox_aspect_ratio
                        bbox.area_percentage = bbox_area_percentage
                        self._valid_targets_bbox_arr.append(bbox)

    def _select_most_viable_bbox(self):
        '''This function selects a bbox from the array of valid target bboxes that is the most viable target.
        The most viable target has the closest match criteria with the prev most viable bbox.

        Args:
            None

        Returns:
            yolo_bbox (bbox): None if no viable target found, else most viable bbox
        '''

        most_viable_target_bbox = None
        total_loss_arr = []

        # Compare each criteria with prev target (self.tracker_bbox): depth, area, dist of centroid from prev bbox
        # The "loss" for each criteria is calculated. The bbox with least sum of all losses is determined to be the
        # closest match.
        for bbox in self._valid_targets_bbox_arr:
            # Get loss in depth
            loss_depth = abs(self.tracker_bbox.depth - bbox.depth)
            
            # Get loss in area
            loss_area = abs(self.tracker_bbox.area_percentage - bbox.area_percentage)

            # Get loss in dist of centroid from prev bbox
            centroid_tracker = [(self.tracker_bbox.xmin + self.tracker_bbox.xmax) // 2, (self.tracker_bbox.ymin + self.tracker_bbox.ymax) // 2]
            centroid_bbox = [(bbox.xmin + bbox.xmax) // 2, (bbox.ymin + bbox.ymax) // 2]
            loss_distance = math.sqrt( ((centroid_tracker[0]-centroid_bbox[0])**2) + 
                                       ((centroid_tracker[1]-centroid_bbox[1])**2) )

            bbox_total_loss = loss_depth + loss_area + loss_distance
            total_loss_arr.append(bbox_total_loss)

        # Pop the bbox with least loss
        most_viable_target_index = total_loss_arr.index(min(total_loss_arr))
        most_viable_target_bbox = self._valid_targets_bbox_arr.pop(most_viable_target_index)

        return most_viable_target_bbox
    
    def tracker_init(self):
        # make lost count zero
        self.tracker_lost_count = 0
        self.target_lost = False
        self.tracker_first_target_found = False

        # Set the value of tracker_bbox to middle of the image, where a person normally stands in front of the robot
        # When first detected bboxes come in, the most likely that matches this criteria will be chosen
        self.tracker_bbox = bbox()
        self.tracker_bbox.xmin = int(self.img_width * (1.0 / 3))
        self.tracker_bbox.ymin = int(self.img_height * (1.0 / 4))
        self.tracker_bbox.xmax = int(self.img_width * (2.0 / 3))
        self.tracker_bbox.ymax = int(self.img_height)
        self.tracker_bbox.depth = 1.0

    def tracker_update(self, input_img, depth_img, bbox_arr_detections):
        '''The tracker's bbox will be centre of image at startup. Target must be inside this area to init tracker
        ''' 
        
        # Create list of valid targets, i.e., bboxes that meet our criteria of a "target"
        self._filter_detected_bboxes(bbox_arr_detections, depth_img)
        
        # See if any valid bboxes are present. If not, return False for failure
        if (len(self._valid_targets_bbox_arr)) == 0:
            if (self.tracker_first_target_found == True):
                self.tracker_lost_count += 1
            if (self.tracker_lost_count > self.tracker_max_lost_count):
                self.target_lost = True
            self.tracker_update_successful = False
        else:
            # Select one of those bboxes as our target
            self.tracker_first_target_found = True
            self.tracker_bbox = self._select_most_viable_bbox()
            self.tracker_lost_count = 0
            self.tracker_update_successful = True
        
        

        return self.tracker_update_successful

    def publish_img_tracker_output(self, input_img):
        # Put info on output img
        # case1: target lost
        # case2: tracking not init yet
        # case3: tracking started, but no valid targets found
        # case4: tracking started, valid targets found
        # case5: unknown behaviour. Raise error.
        rgb_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

        if (self.target_lost == True):
            text_origin = (30,30) #(self.img_width + 50, self.img_height + 50)
            label = 'No viable target found for {} frames'.format(self.tracker_max_lost_count)
            cv2.putText(rgb_img, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            text_origin = (100,150) #(self.img_width + 50, self.img_height + 50)
            label = 'Target Lost'
            cv2.putText(rgb_img, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 0), 2)
            text_origin = (100,200) #(self.img_width + 50, self.img_height + 50)
            label = 'Restart Program'
            cv2.putText(rgb_img, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 0), 2)

        
        elif (self.tracker_first_target_found == False):
            left, right = self.tracker_bbox.xmin, self.tracker_bbox.xmax
            top, bottom = self.tracker_bbox.ymin, self.tracker_bbox.ymax
            text_color = (0, 0, 255)
            cv2.rectangle(rgb_img, (left, top), (right, bottom), text_color, 2)

            label = 'Waiting to acquire Target...'
            text_origin = (left + 5, top + 25)
            cv2.putText(rgb_img, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)
        
        elif (self.tracker_update_successful == False):
            text_color = (255, 0, 0)
            label = 'No Viable Target Found'
            text_origin = (30,30) #(self.img_width + 50, self.img_height + 50)
            cv2.putText(rgb_img, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)
        
        elif (self.tracker_update_successful == True):
            # Add all valid bboxes in red
            for bbox in self._valid_targets_bbox_arr:
                # Add rectangles with label to detected objects in image for visualization
                left, right = bbox.xmin, bbox.xmax
                top, bottom = bbox.ymin, bbox.ymax
                cv2.rectangle(rgb_img, (left, top), (right, bottom), (255, 0, 0), 2)
                text_origin = (left + 5, top + 25)
                label = '{}, {:.2f}'.format(bbox.Class, bbox.prob)
                cv2.putText(rgb_img, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

                # Add depth info to center of each bbox
                centroid_x = (left + right) / 2
                centroid_y = (top + bottom) / 2
                radius = 5
                cv2.circle(rgb_img, (centroid_x, centroid_y), radius, (255, 255, 255), -1)
                text_origin = (centroid_x - 15, centroid_y - 15)
                label = '{:.2f}m'.format(bbox.depth)
                cv2.putText(rgb_img, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            
            # Add selected target bbox in green
            text_color = (0, 255, 0)
            label = '{}, {:.2f}'.format(self.tracker_bbox.Class, self.tracker_bbox.prob)
            left, right = self.tracker_bbox.xmin, self.tracker_bbox.xmax
            top, bottom = self.tracker_bbox.ymin, self.tracker_bbox.ymax
            cv2.rectangle(rgb_img, (left, top), (right, bottom), text_color, 2)
            text_origin = (left + 5, top + 25)
            cv2.putText(rgb_img, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)
            # Add depth info of target bbox
            centroid_x = (left + right) // 2
            centroid_y = (top + bottom) // 2
            radius = 5
            cv2.circle(rgb_img, (centroid_x, centroid_y), radius, (255, 255, 255), -1)
            text_origin = (centroid_x - 15, centroid_y - 15)
            label = '{:.2f}m'.format(self.tracker_bbox.depth)
            cv2.putText(rgb_img, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        else:
            rospy.loginfo('Case5')
            raise ValueError('Unknown behaviour. Case5, check comment above.')

        # Publish img labeled with valid bboxes
        labelled_img_ros = self.bridge.cv2_to_imgmsg(rgb_img, encoding="rgb8")
        self.tracker_labelled_img_pub.publish(labelled_img_ros)

    

        
def main(args):
    rospy.init_node('sdroller_track_detect_node')

    yolo = ObjectDetectionTracking()

    start_time = time.time()
    counter = 0

    while (not rospy.is_shutdown()):
        if yolo.new_img_received:
            # Check if new img received from camera. If so, create copy of current img so it's not overwritten by new img.
            yolo.new_img_received = False
            input_img = np.copy(yolo.rgb_img)
            depth_img = np.copy(yolo.depth_img)
            
            # Get list of bboxes of objects detected in img
            bbox_arr_detections = yolo.classify(input_img)

            # Select a target
            ok = yolo.tracker_update(input_img, depth_img, bbox_arr_detections)
            yolo.publish_img_tracker_output(input_img)

            if (yolo.target_lost == False):
                # Selected target is stored in yolo.tracker_bbox
                if (ok):
                    str_info = ('Target Found! \t bbox: [({}, {}), ({}, {})] \ndepth: {}, area: {}, aspect ratio: {}'.format(
                                yolo.tracker_bbox.xmin, yolo.tracker_bbox.ymin,
                                yolo.tracker_bbox.xmax, yolo.tracker_bbox.ymax,
                                yolo.tracker_bbox.depth, yolo.tracker_bbox.area_percentage,
                                yolo.tracker_bbox.aspect_ratio))
                    rospy.loginfo(str_info)
                else:
                    rospy.loginfo('No Valid Target Detected')
            else:
                str_info = '\n  No viable target has been found for {} frames.\n  Target is deemed to be lost. Restart Program'.format(yolo.tracker_max_lost_count)
                rospy.logerr(str_info)

if __name__ == '__main__':
    main(sys.argv)

'''
how to store the depth of each bbox? create entry within bbox datatype? hmm. alternately, create an array of properties, just as we create an array of bboxes.
or have multidimensional array with bboxes and properties.
or have dict with each entry an array (all indexes match)

Need to check for depth camera giving incorrect depth of person (0, inf, nan). Median depth will give incorrect results if person too close/invalid. 
Centre of bbox (current method) also not very reliable (person can spread hands while sideways). But easy to show in labelled img. put dot in centre of img 
with value of depth next to dot. Can see where the depth coming from and when errors out.
'''