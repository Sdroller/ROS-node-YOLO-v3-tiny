
# sdroller_track_detect
This repo implements object detection and tracking. YOLOv3 or YOLOv3-tiny is used for object detection. 
The tracking is based on selecting the most viable bounding box based on criteria like distance to target, size of bounding box, etc.

There are several tunable parameters, they can be found in config/object_detection.yaml:

```yaml
tracker:
  target_class        : 'person'  # The class to be tracked.
  tracker_init_min_iou: 0.2       # Min IoU of target class with tracker's bbox. This is used for first initialization of tracker only
  threshold_confidence: 0.4       # Min confidence of a detection for it to be considered a valid target
  target_max_dist     : 2.0       # Max dist of target from camera in meters for it to be considered a valid target
```

### How it Works
For every image, Yolo is run to obtain bounding boxes (bboxes) of detected objects.
Viable targets are selected from this list based on several criteria:

- depth to target
- aspect ratio of bbox
- area of bbox as a percentage of area of img
- class of object detected

Out of these viable targets, the one bbox which most closely resembles the target is chosen. Warning: rejection
of a viable bbox based on the fact that it's depth/location is too far from prev selected target is not done. See
explanation given later below.

If a valid target is not found for N consequtive frames, then the target is deemed to be lost. Put the brakes on,
lock the baby and alert emergency contacts about a lost parent!


#### Calculation of Depth

The depth is calculated as the mean value of 10x10 pixels around the centroid of a bounding box. We cannot use
metrics like median or mean directly in case the target's depth is not available from camera, i.e., the target's
depth is marked as 0, NaN, +/- inf. This can happen when the target is too close/reflective/etc.

#### Guide to choosing aspect ratio limits

Aspect ratio is calculated as img_width/img_height. It can define the relative shape of a bounding box. 
An idea of diff aspect ratios of bounding box for people can be gained from the paper ["SmartMonitor"-An Intelligent Security System for the Protection of Individuals and Small Properties with the Possibility of Home Automation](https://www.researchgate.net/figure/Aspect-ratios-of-bounding-boxes-created-for-different-human-silhouettes-detected-in-the_tbl1_262930297), Table 2.

#### Rejection of viable bboxes w.r.t. prev bbox

When tracker is selecting a target, it does not reject bboxes based on the diff of criteria from prev selected bbox.
Meaning: Suppose tracker is latched onto a person that is 0.5m away and has bbox at location (100,100).  
If there is only 1 bbox detected at next frame, and it has a person that is 2.0m away and bbox is located at (288,288),
the tracker will not reject this new bbox because the depth is suddenly so different and bbox itself is far away from its
prev location.

The reasoning is that when running, sudden movements can occur that may change the depth/location of bbox by large values.
Also, there might possibly be a lag in the system which would result in large change from prev bbox to current list of bboxes.

Therefore, during initialization, the robot will latch onto the first viable target seen. If multiple target are present during
initialization, it will select the most viable target based on criteria like depth, area and dist to prev bbox.

## Instructions to Setup

Clone all required repositories into the src directory of a catkin workspace:

```bash
~/catkin-ws/src$ git clone git@github.com:Sdroller/sdroller_bringup.git
~/catkin-ws/src$ git clone git@github.com:Sdroller/zed-ros-wrapper.git
~/catkin-ws/src$ git clone git@github.com:Sdroller/sdroller_track_detect.git
```

Use the `tracking-test` branch of `sdroller_bringup` and  
Use the `criteria_selection_tracker` branch of `sdroller_track_detect`:
```bash
$ cd sdroller_bringup
$ git checkout tracking-test
$ cd sdroller_track_detect
$ git checkout criteria_selection_tracker
```

Build the workspace: `$ catkin build`

```bash
# Optional: blacklist unrequired zed packages to speed up build:
$ catkin config --blacklist zed_nodelet_example zed_rtabmap_example zed_depth_sub_tutorial zed_tracking_sub_tutorial zed_video_sub_tutorial

$ catkin build
```

## Instructions to run

To run this repo, simply run the bringup launch file. This will launch the zed camera, yolo+tracker and 
also 2 windows to show the output of yolo and tracker.  
Note that it takes ~2 min for the weights to be loaded and detection to start on the Jetson TX2:

```bash
$ roslaunch sdroller_bringup sdroller_bringup.launch run_tracker:=true view_output:=true
```

.  
.  
.  
#################### ORIGINAL README OF ROS-node-YOLO-v3-tiny ####################
# YOLOv3-tiny on Jetson tx2

This is a tested ROS node for YOLOv3-tiny on Jetson tx2.

Please see the medium post to get the understanding about this repo: https://medium.com/intro-to-artificial-intelligence/run-yolo-v3-as-ros-node-on-jetson-tx2-without-tensorrt-43f562aadc68

### Credit

I have used components of below resources to make this ROS node. Thanking them for the great effort.

* https://medium.com/@manivannan_data/how-to-train-yolov3-to-detect-custom-objects-ccbcafeb13d2
* https://medium.com/@manivannan_data/how-to-train-yolov2-to-detect-custom-objects-9010df784f36
* https://github.com/qqwweee/keras-yolo3
