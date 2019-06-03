cmake_minimum_required(VERSION 2.8.3)
project(project_filter_icp)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
set(PACKAGE_DEPENDENCIES people_detector)
set(PCL_DIR /home/lhd/Documents/pcl-1.9/share/pcl-1.9)

set(TERGEO_ROOT_PATH /home/lhd/tongji/tergeo)
include_directories(${TERGEO_ROOT_PATH})
message(${TERGEO_ROOT_PATH}) 
include(${TERGEO_ROOT_PATH}/cmake/3rdparty.cmake)
include(${TERGEO_ROOT_PATH}/cmake/options.cmake)
include(${TERGEO_ROOT_PATH}/cmake/find_tergeo.cmake)
include(${TERGEO_ROOT_PATH}/cmake/find_tergeo_algorithm.cmake)
include(${TERGEO_ROOT_PATH}/cmake/find_qt.cmake)
include(${TERGEO_ROOT_PATH}/cmake/find_vtk.cmake)

find_package(PCL REQUIRED COMPONENTS)
find_package(OpenCV REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  pcl_conversions
  pcl_msgs
  pcl_ros
  roscpp
  sensor_msgs   
  detector_msg 
)
catkin_package(
 INCLUDE_DIRS include
 LIBRARIES project_filter_icp
 CATKIN_DEPENDS pcl_conversions pcl_msgs pcl_ros roscpp sensor_msgs message_runtime
#  DEPENDS system_lib
)

include_directories(
  include
  /usr/include/eigen3/
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)
link_directories(${PCL_LIBRARY_DIRS})

add_executable(${PROJECT_NAME} src/merge.cpp)
add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})## target back to the shorter version for ease of user use

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${roslib_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
)

