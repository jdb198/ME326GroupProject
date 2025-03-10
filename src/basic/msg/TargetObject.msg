int32 x
int32 y
uint8 axis         # 0: x, 1: y (only used when purpose is to grasp the object)
uint8 purpose      # 0: Navigation, 1: Grasping

geometry_msgs/PoseStamped pose  # Full pose of the target object
