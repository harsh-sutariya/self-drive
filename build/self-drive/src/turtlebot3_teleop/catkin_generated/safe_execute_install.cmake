execute_process(COMMAND "/home/ai4ce/team3-deploy/catkin_s25/build/self-drive/src/turtlebot3_teleop/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/home/ai4ce/team3-deploy/catkin_s25/build/self-drive/src/turtlebot3_teleop/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
