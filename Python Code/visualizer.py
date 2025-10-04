
import numpy as np
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Pose
from visualization_msgs.msg import Marker, MarkerArray
from .robot_arm import Edubot

#This is the answer to part 1.2 for the AE 4324 
"""
Using your visualization software of choice (Python, Matlab, ROS2+RVIZ, etc.), make
a 3D visualization of the robot workspace if
3. None of the robot joints would be constrained.
4. The actual joint limits apply.
"""

#The method used is to calculate forward kinematics over a range of motion for each link
class Visualizer(Node):
    def __init__(self):
        super().__init__("visualizer")
        self.pub = self.create_publisher(MarkerArray, 'workspace_markers', 10)
        
        #Instantiate Robot 
        self.robot = Edubot()

        #Create Marker array to hold all marker points
        self.samplePoints = 15
        self.num_markers = self.samplePoints**4
        self.marker_array = MarkerArray()

        self.plot_workspace()

    def plot_workspace(self):
        #Could use itertools here instead of nested loops
        [q0Bounds, q1Bounds, q2Bounds, q3Bounds] = self.robot.get_bounds()
        j0_space = np.linspace(q0Bounds[0], q0Bounds[1], 3)
        idx = 0
        for j0 in j0_space: 
            j1_space = np.linspace(q1Bounds[0], q1Bounds[1], self.samplePoints)
            for j1 in j1_space: 
                j2_space = np.linspace(q2Bounds[0], q2Bounds[1], self.samplePoints)
                for j2 in j2_space: 
                    j3_space = np.linspace(q3Bounds[0], q3Bounds[1], self.samplePoints)
                    for j3 in j3_space:
                        angles = [j0, j1, j2, j3]
                        point_i = self.robot.forward_kinematics(angles)

                        #Compile point into array to be plotted with pointCloud2 
                        pose = Pose()
                        pose.position.x = point_i[0]
                        pose.position.y = point_i[1]
                        pose.position.z = point_i[2]
                        pose.orientation.w = 1.0 #arbitrary

                        #Create Marker
                        marker = Marker()
                        marker.header.frame_id = "base"
                        marker.id = idx
                        marker.type = Marker.SPHERE
                        marker.scale.x = 0.01
                        marker.scale.y = 0.01
                        marker.scale.z = 0.01
                        marker.color.a = 1.0 #opacity
                        marker.color.g = 1.0 #Make the marker white
                        marker.color.r = 1.0 #Make the marker white
                        marker.color.b = 1.0 #Make the marker white
                        marker.pose = pose

                        #Add marker to marker_array(consider making marker_array fully allocated b)
                        self.marker_array.markers.append(marker)
                        idx += 1
        
        self.pub.publish(self.marker_array)

def main():
    rclpy.init()
    node = Visualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
if __name__ == "__main__":
    main()

