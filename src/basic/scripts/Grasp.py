import math
from interbotix_xs_modules.xs_robot.locobot import InterbotixLocobotXS
import numpy as np
import time

# THIS SCRIPT DOESN'T WORK WITHOUT FIXING THE PUBLISHERS AND SUBSCRIBERS. 
# ALSO PUT IN THE CALL FOR THE FINGER DISTANCE 

def main():
    
    locobot = InterbotixLocobotXS(robot_model="locobot_wx250s", arm_model="mobile_wx250s")

    locobot.arm.go_to_home_pose()
    locobot.gripper.release()
    grasp_complete = False
    while grasp_complete == False:
        #####get_coordinates = # subscribe and get coordinates from image
        locobot.arm.set_ee_pose_components(x=x, y=y, z=z, roll=1.0, pitch=1.5)

        locobot.gripper.grasp()
        locobot.arm.go_to_home_pose()
        #locobot.arm.go_to_sleep_pose()
        #finger_distance # check for grasp.
        if finger_grasp > np.abs(.002):
            grasp_complete = True
        else:
            time.sleep(3)


if __name__=='__main__':
    main()
