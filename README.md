
# sdroller_track_detect
This repo implements object detection and tracking. YOLOv3-tiny is used for object detection. OpenCV trackers KCF or MOSSE are used for tracking. The target class to be tracked is "person". On a TX2, running in max mode, detection runs at ~11fps, but is triggered only once per second. Detection with MOSSE tracker can run at 40+fps.

There are several tunable parameters, like:
```
        self.target_class = 'person'  # The class to be tracked.
        self.tracker_min_iou = 0.3  # Min value of IoU for a bbox to be considered same as tracked object.
        self.tracker_init_min_iou = 0.2  # This is used for first initialization of tracker only
        self.threshold_confidence = 0.4  # prob of class must be greater than this threshold for us to use it.
```

## Instructions to run
Use the `tracking-test` branch of `sdroller_bringup`:
```
$ cd sdroller_bringup
$ git checkout tracking-test
```

To run this repo, make sure zed camera is publishing rgb and depth. 
```
$ roslaunch sdroller_bringup sdroller_bringup.launch
```
Then start this repo. Note that it takes ~2 min for the weights to be loaded and detection to start:
```
$ roslaunch sdroller_track_detect sdroller_track_detect.launch
```



########## ORIGINAL README OF ROS-node-YOLO-v3-tiny ##########
# YOLOv3-tiny on Jetson tx2

This is a tested ROS node for YOLOv3-tiny on Jetson tx2.

Please see the medium post to get the understanding about this repo: https://medium.com/intro-to-artificial-intelligence/run-yolo-v3-as-ros-node-on-jetson-tx2-without-tensorrt-43f562aadc68

### Credit

I have used components of below resources to make this ROS node. Thanking them for the great effort.

* https://medium.com/@manivannan_data/how-to-train-yolov3-to-detect-custom-objects-ccbcafeb13d2
* https://medium.com/@manivannan_data/how-to-train-yolov2-to-detect-custom-objects-9010df784f36
* https://github.com/qqwweee/keras-yolo3
