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
import time 

class FollowTraj(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        #General Concept: At set interval in time, iterate through a series of points 
        #Load in edubot
        self.robot = Edubot()

        #load trajectory parameters
        self.idx = 0 #To keep up with which of the points we're currently at
        self._beginning = self.get_clock().now()
        self.timer_period = 0.1  # [s] represents how often a new point is moved to
        self.trajectory_period = 15 # [s] represents how much time it will take to complete the trajectory
        self.traj_array = self.get_trajectory()

        #Create publisher and start timer that runs through trajectory
        #Create publisher and start timer that runs through trajectory
        self._publisher = self.create_publisher(JointTrajectory, 'joint_cmds', 10)
        self.marker_pub = self.create_publisher(Marker, 'traj_markers', 10)
        self._timer = self.create_timer(self.timer_period, self.continue_trajectory)

    def get_trajectory(self):
        #List and select the avaliable files within the working directory
        cwd = "/home/henry/ros2_jazzy/edubot/python_impl/src/controllers/controllers/"
        csv_files = [file for file in os.listdir(cwd) if file.lower().endswith('.csv')]
        filenames = csv_files
        print(f"Here are the csv files avaliable to run as trajectories within {cwd}:")
        for idx, file in enumerate(filenames):
            print(f"{idx}: {file}\n")
        
        #Choose which file you want
        filename = cwd + filenames[int(input("Please select which number you want to run: "))]
        print(f"Thanks! Creating trajectory for {filename} with a period of {self.trajectory_period}s, updating every {self.timer_period}s\n")
        
        #Gather file and cut into the right size 
        num_points = math.floor(self.trajectory_period/self.timer_period) #[s] total # of points to iterate through(round down)
        data = pd.read_csv(filename)
        trajectory = np.array([data.q0, data.q1, data.q2, data.q3]).T
        every_x_points = math.floor(np.size(trajectory, 0) / num_points) #Pull points at this interval
        zero_indices = np.where(np.arange(np.size(trajectory, 0)) % every_x_points == 0)
        reduced_points = trajectory[zero_indices, :]
        #squeeze the array to remove the unecessary axis and change from dim of (1, x, 4) to (x, 4) 
        return np.squeeze(reduced_points, axis=0)
    
    def continue_trajectory(self):
        #Work through the points, publishing as you go along
        #print(np.shape(self.traj_array))
        joint_i = self.traj_array[self.idx, :]
        #print(np.shape(joint_i))
        #publish trajectory point
        now = self.get_clock().now()
        msg = JointTrajectory()
        msg.header.stamp = now.to_msg()

        point = JointTrajectoryPoint()
        #print(joint_i)
        point.positions = [joint_i[0].item(), joint_i[1].item(), joint_i[2].item(), joint_i[3].item(), 0.5]
        msg.points = [point]
        self._publisher.publish(msg)

        #publish marker
        #First get point with forward kinematics
        point_i, _ = self.robot.forward_kinematics(joint_i)

        #Compile point into array to be plotted with pointCloud2 
        pose = Pose()
        pose.position.x = point_i[0]
        pose.position.y = point_i[1]
        pose.position.z = point_i[2]
        pose.orientation.w = 1.0 #arbitrary
        
        #Create Marker
        marker = Marker()
        marker.header.frame_id = "base"
        marker.id = self.idx
        marker.type = Marker.SPHERE
        marker.scale.x = 0.01
        marker.scale.y = 0.01
        marker.scale.z = 0.01
        marker.color.a = 1.0 #opacity
        marker.color.g = 1.0 #Make the marker white
        marker.color.r = 0.0 #Make the marker white
        marker.color.b = 1.0 #Make the marker white
        marker.pose = pose

        #Add marker to marker_array(consider making marker_array fully allocated b)
        self.marker_pub.publish(marker)
        
        if self.idx < np.size(self.traj_array, 0)-1:
            #if it hasn't reached the end of the array, keep going
            self.idx += 1
        else:  
            #if it has reached the end of the array, go slowly back to old position, redraw
            final_jpos = self.traj_array[0, :]
            starting_jpos = self.traj_array[-1, :]
            print(f"trajectory ended. Moving from {starting_jpos} to {final_jpos} before restarting") 
            self.move_j(final_jpos, starting_jpos, 5, 50)
            self.idx = 0
        return 
        
    def move_j(self, final_joints, starting_joints, traj_time, num_points):
        #Takes in final joints for q0 to q3, not the gripper
        wait_time = traj_time / num_points
        print(f"Final joint array: {final_joints}, plus {starting_joints}")
        final_joints = np.append(final_joints, 0.5)
        starting_joints = np.append(starting_joints, 0.5)
        interpolated_trajectory = np.array([np.linspace(starting_joints[i], final_joints[i], num_points) for i in range(len(final_joints))]).T
        
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
