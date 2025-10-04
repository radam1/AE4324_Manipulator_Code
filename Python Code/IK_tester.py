import rclpy
import math
import numpy as np
from rclpy.node import Node
from .robot_arm import Edubot
import os
import pandas as pd
from geometry_msgs.msg import Pose
from visualization_msgs.msg import Marker
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from pynput import keyboard
import time

class FollowTraj(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        #General Concept: At set interval in time, iterate through a series of points 
        #Load in edubot
        self.robot = Edubot()
        self.joint_home = np.array([0, 0, 0, 0, 0.5])
        self.current_joints = self.joint_home.copy() #Start at home position

        #load trajectory parameters
        self._beginning = self.get_clock().now()

        #For the gripper control
        self.gripper_active = False
        self.keys_pressed = {}
        def on_press(key):
            try:
                self.keys_pressed[key.char] = True
            except AttributeError:
                # Special keys
                if key == keyboard.Key.up:
                    self.keys_pressed['up'] = True
                elif key == keyboard.Key.down:
                    self.keys_pressed['down'] = True
                elif key == keyboard.Key.left:
                    self.keys_pressed['left'] = True
                elif key == keyboard.Key.right:
                    self.keys_pressed['right'] = True
        
        def on_release(key):
            try:
                self.keys_pressed[key.char] = False
            except AttributeError:
                # Special keys
                if key == keyboard.Key.up:
                    self.keys_pressed['up'] = False
                elif key == keyboard.Key.down:
                    self.keys_pressed['down'] = False
                elif key == keyboard.Key.left:
                    self.keys_pressed['left'] = False
                elif key == keyboard.Key.right:
                    self.keys_pressed['right'] = False
        
        # Start keyboard listener
        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()

        #Create publisher and start timer that runs through trajectory
        self._publisher = self.create_publisher(JointTrajectory, 'joint_cmds', 10)
        self.marker_pub = self.create_publisher(Marker, 'traj_markers', 10)      

        #Laying out the process of the pick-and-place task: 
        #Step 0: Start at Home Position
        self.go_home()

        #Step 1: Move to the position of the blueberry 
        point1 = np.array([0.1, 0.1, 0.1])
        joint_state_1, _ = self.robot.inverse_kinematics_newton_raphson(point1)
        point2 = np.array([0.2, 0.1, 0.3])
        joint_state_2, _ = self.robot.inverse_kinematics_newton_raphson(point2)
        point3 = np.array([0.0, 0.0, 0.3])
        joint_state_3, _ = self.robot.inverse_kinematics_newton_raphson(point3)
        point4 = np.array([0.0, 0.0, 0.07])
        joint_state_4, _ = self.robot.inverse_kinematics_newton_raphson(point4)
        
        if bool(input("Type 0 for move_j or 1 for move_l")):
            move_j = True
            move_l = False
        else:
            move_j = False
            move_l = True
        
        if move_j:
            self.move_j(joint_state_1, 10, 100)
        if move_l:
            self.move_l(point1, 10, 100)
        #Step 2: Manually close the gripper to not crush the blueberry
        self.gripper_control()

        #Step 3: Move to point 2
        if move_j:
            self.move_j(joint_state_2, 10, 100)
        if move_l:
            self.move_l(point2, 10, 100)

        #Step 4: Manually open the gripper to drop the blueberry in the cup
        self.gripper_control()

        #Step 5: Move to point 3
        if move_j:
            self.move_j(joint_state_3, 10, 100)
        if move_l:
            self.move_l(point3, 10, 100)
        self.gripper_control()

        self.move_j(joint_state_4, 10, 100)
        self.gripper_control()
        #Step 7: Go back home as a midpoint
        self.go_home(self.current_joints)
        
    def go_home(self, starting_joints=None):
        #take the robot to the home position. If there is a starting position provided, move 
        #smoothly to the home position via interpolation with move_l
        if np.any(starting_joints):
            self.move_j(self.joint_home[:4], 5, 50)
        else:
            #If no starting point provided, just publish the home position. Robot will snap to home, but that should be ok(hopefully) 
            now = self.get_clock().now()
            msg = JointTrajectory()
            msg.header.stamp = now.to_msg()

            point = JointTrajectoryPoint()
            point.positions = self.joint_home
            msg.points = [point]
            self._publisher.publish(msg)
        return 
    
    def gripper_control(self):
        print("Now actively controlling gripper. Press a and d to control gripper and q to move on")
        self.gripper_active = True
        change_gripper_by = 0.05 #rad
        period = 0.05 #s
        
        # Small delay to ensure listener is active
        time.sleep(0.1)

        while self.gripper_active:
            # Reset gripper_change for each iteration
            gripper_change = 0
            
            # Control gripper with a/d keys
            if self.keys_pressed.get('d', False):
                gripper_change += change_gripper_by #rad
            if self.keys_pressed.get('a', False):
                gripper_change -= change_gripper_by #rad

            #make sure that the gripper cannot be commanded to overactuate
            if self.current_joints[4] + gripper_change > np.pi/2 or self.current_joints[4] + gripper_change < -np.pi/2:
                gripper_change = 0
            
            self.current_joints[4] += gripper_change

            print(f"Current value = {np.round(self.current_joints[4], 4)}", end="\r")

            #Check for the quit message
            if self.keys_pressed.get('q', False):
                self.gripper_active = False

            #now publish the current joint with updated gripper as the new position
            now = self.get_clock().now()
            msg = JointTrajectory()
            msg.header.stamp = now.to_msg()

            point = JointTrajectoryPoint()
            point.positions = self.current_joints.copy()
            msg.points = [point]
            self._publisher.publish(msg)

            time.sleep(period)
        
        print("Gripper control has been quit. Moving on...")
        return 
    
    def move_j(self, final_joints, traj_time, num_points):
        #Takes in final joints for q0 to q3, not the gripper
        wait_time = traj_time / num_points
        print(f"Final joint array: {final_joints}, plus {self.current_joints[4]}")
        final_joints = np.append(final_joints, self.current_joints[4])
        interpolated_trajectory = np.array([np.linspace(self.current_joints[i], final_joints[i], num_points) for i in range(len(final_joints))]).T
        
        for joint_state in interpolated_trajectory:
            #Publish the point in the current trajectory
            now = self.get_clock().now()
            msg = JointTrajectory()
            msg.header.stamp = now.to_msg()

            point = JointTrajectoryPoint()
            point.positions = joint_state
            msg.points = [point]
            self._publisher.publish(msg)

            #Wait for the joint so it doesn't send them all at once
            time.sleep(wait_time)
        print("Move in J completed. Moving on...")
        self.current_joints = final_joints
        return 
    
    def move_l(self, final_pos, traj_time, num_points, final_gripper=None):
        wait_time = traj_time/num_points
        current_pos = self.robot.forward_kinematics(self.current_joints[:4])
        interpolated_trajectory = np.array([np.linspace(current_pos[i], final_pos[i], num_points) for i in range(len(final_pos))]).T
        joint_array = np.zeros((np.size(interpolated_trajectory, 0), 5))
        
        for idx, point in enumerate(interpolated_trajectory):
            joint_array[idx, :4] = self.robot.inverse_kinematics_newton_raphson(point)
            joint_array[idx, 4] = self.current_joints[4]
        
        for joint_state in joint_array:
            #Publish the point in the current trajectory
            now = self.get_clock().now()
            msg = JointTrajectory()
            msg.header.stamp = now.to_msg()

            point = JointTrajectoryPoint()
            point.positions = joint_state
            msg.points = [point]
            self._publisher.publish(msg)

            #Wait for the joint so it doesn't send them all at once
            time.sleep(wait_time)
        print("Move in L completed. Moving on...")
        self.current_joints = joint_array[-1,:]
        return 
    
def main(args=None):
    rclpy.init(args=args)

    example_traj = FollowTraj()

    rclpy.spin(example_traj)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    example_traj.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
