cmake_minimum_required(VERSION 2.8.3)
project(detector_msg)

# add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
)


## Generate messages
add_message_files(
  FILES
    bounding_box.msg
    detection_result.msg
)
## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS
  message_runtime
  std_msgs
)