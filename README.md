# ME326GroupProject

Welcome to our Repository! Please visit the final_code branch and navigate to /src/basic/scripts to check out our Python and ROS2 node setups and code.

Inside our scripts folder, please reference the audio_recorder_node.py to see how we implemented the algorithm to record a 5-second command. Please refer to the audio_transcriber_node.py file to see how we implemented the algorithm to analyze the recorded message. In the perception_node.py, the algorithms detect the desired object based on the given command and publishes the coordinate relative to the target reference frame. The navigation_node.py file has our algorithm to navigate the robot just in front of the object. Finally, the manipulation_node.py files are responsible for controlling the robotic arm and moving the end effector to where it needs to go, based on the feedback from the perception node.
