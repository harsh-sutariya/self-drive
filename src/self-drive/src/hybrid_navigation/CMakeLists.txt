cmake_minimum_required(VERSION 3.0.2)
project(hybrid_navigation)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  cv_bridge
  message_generation
)

## Generate services in the 'srv' folder
add_service_files(
  FILES
  LoadTarget.srv
)

## Generate added messages and services
generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS 
  rospy 
  std_msgs 
  geometry_msgs 
  nav_msgs 
  sensor_msgs
  cv_bridge
  message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/topological_mapper.py
  scripts/visual_place_recognition.py
  scripts/hierarchical_navigator.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
) 