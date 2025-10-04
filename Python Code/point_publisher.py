import rclpy
from .robot_arm import Edubot
import numpy as np
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class ExampleTraj(Node):

    def __init__(self):
        super().__init__('minimal_publisher')

        self.robot = Edubot()
        self._publisher = self.create_publisher(JointTrajectory, 'joint_cmds', 10)
        self.publish_point()

    def publish_point(self):
        now = self.get_clock().now()
        msg = JointTrajectory()
        msg.header.stamp = now.to_msg()
        input_string = input("Put in your desired point here: x,y,z\n")
        input_list = [float(i) for i in input_string.split(",")]
        desired_position, _ = self.robot.inverse_kinematics_newton_raphson(input_list)
        
        point = JointTrajectoryPoint()
        point.positions = [desired_position[0],
                           desired_position[1],
                           desired_position[2],
                           desired_position[3],
                           0.0
                           ]
        msg.points = [point]

        self._publisher.publish(msg)
        

def main(args=None):
    rclpy.init(args=args)

    example_traj = ExampleTraj()

    rclpy.spin(example_traj)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    example_traj.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
