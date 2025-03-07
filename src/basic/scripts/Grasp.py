import math
from interbotix_xs_modules.xs_robot.locobot import InterbotixLocobotXS

# This script makes the end-effector perform pick, pour, and place tasks
#
# To get started, open a terminal and type...
# 'roslaunch interbotix_xslocobot_control xslocobot_python.launch robot_model:=locobot_wx250s show_lidar:=true'
# Then change to this directory and type 'python bartender.py'

def main():
    joint_positions = [-1.0, 0.5, 0.5, 0, -0.5, 1.57]
    locobot = InterbotixLocobotXS(robot_model="locobot_wx250s", arm_model="mobile_wx250s")
    # locobot.arm.set_ee_pose_components(x=0.1, z=0.2)
    # locobot.arm.set_single_joint_position("waist", math.pi/4.0)
    locobot.arm.go_to_home_pose()
    # locobot.arm.set_single_joint_position("waist", math.pi/6.0)
    locobot.arm.set_ee_pose_components(x=0.2, y=0.1, z=0.2, roll=1.0, pitch=1.5)
    # locobot.arm.set_joint_positions(joint_positions)

    locobot.gripper.release()
    #locobot.arm.set_ee_cartesian_trajectory(x=0.01, z=0.25)
    locobot.gripper.grasp()
    #locobot.arm.set_ee_cartesian_trajectory(x=-0.1, z=0.25)
    #locobot.arm.set_single_joint_position("waist", -math.pi/4.0)
    #locobot.arm.set_ee_cartesian_trajectory(pitch=1.5)
    #locobot.arm.set_ee_cartesian_trajectory(pitch=-1.5)
    #locobot.arm.set_single_joint_position("waist", math.pi/4.0)
    #locobot.arm.set_ee_cartesian_trajectory(x=0.1, z=-0.25)
    #locobot.gripper.open()
    #locobot.arm.set_ee_cartesian_trajectory(x=-0.1, z=0.25)
    locobot.arm.go_to_home_pose()
    locobot.arm.go_to_sleep_pose()

if __name__=='__main__':
    main()