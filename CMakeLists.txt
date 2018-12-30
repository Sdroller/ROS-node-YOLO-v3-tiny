cmake_minimum_required(VERSION 2.8.3)
project(sdroller_track_detect)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  cv_bridge
  image_transport
  message_generation
)

find_package(OpenCV)


add_message_files(
  FILES
  bbox.msg
  bbox_array.msg
)

generate_messages(
   DEPENDENCIES
   std_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy message_runtime
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

