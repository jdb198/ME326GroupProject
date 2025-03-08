#!/usr/bin/env python3
import math
from interbotix_xs_modules.xs_robot.locobot import InterbotixLocobotXS
import numpy as np
import time

# THIS SCRIPT DOESN'T WORK WITHOUT FIXING THE PUBLISHERS AND SUBSCRIBERS. 
# ALSO PUT IN THE CALL FOR THE FINGER DISTANCE 

def main():
    grasp = 0
    locobot = InterbotixLocobotXS(robot_model="locobot_wx250s", arm_model="mobile_wx250s")

    

    while grasp == 0: 
        locobot.arm.go_to_home_pose()
        time.sleep(3)
        locobot.gripper.release()
        locobot.arm.set_ee_pose_components(x=.5, y=.02, z= -.03, roll=1.0, pitch=1.5)
        time.sleep(3)
        locobot.gripper.grasp()
        locobot.arm.go_to_home_pose()
        time.sleep(5)
        finger_position = locobot.gripper.get_finger_position()
        print(finger_position)

        if finger_position > .02: # this is telling us if it is grasping or not. 
            print('Success')
            grasp = 1
            break
        else:
            print('Failure')
            grasp = 0

    locobot.gripper.release()
    locobot.arm.go_to_sleep_pose()



if __name__=='__main__':
    main()
