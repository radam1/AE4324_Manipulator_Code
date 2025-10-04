import rclpy
import numpy as np
from rclpy.node import Node
from .robot_arm import Edubot
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from pynput import keyboard

class ExampleTraj(Node):

    def __init__(self):
        super().__init__('minimal_publisher')

        self.robot = Edubot()
        self._HOME = [0, 0, 0, 0, 0.5] #Home joint array(with gripper)
        self.initial_position, _ = self.robot.forward_kinematics(np.array(self._HOME)) #Home cartesian position of EE
        self._beginning = self.get_clock().now()
        self._publisher = self.create_publisher(JointTrajectory, 'joint_cmds', 10)
        timer_period = 0.05  #[s] seconds between sampling 
        desired_velocity = 0.05 #[m/s] corresponds to 0.5cm per second.  
        self.delta_pos = desired_velocity * timer_period
        self._timer = self.create_timer(timer_period, self.timer_callback)
        
        # Initialize current position and joint array
        self.current_position = self.initial_position.copy()
        self.current_joints = self._HOME.copy()
        
        # Set up key tracking dictionary
        self.keys_pressed = {}
        
        # Initialize keyboard listener to update dictionary
        try:
            
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
            
        except ImportError:
            self.get_logger().error("pynput not installed. Install with: pip install pynput")
    
    def timer_callback(self):
        # Calculate differences based on key presses
        relative_change = np.zeros((3, 1))  # For 3 axes 
        gripper_change = 0
        change_gripper_by = 0.05 #rad
        max_joint_move = 0.05 #rad 

        # Control x-axis with w/s keys
        if self.keys_pressed.get('w', False):
            relative_change[0] -= self.delta_pos #m
        if self.keys_pressed.get('s', False):
            relative_change[0] += self.delta_pos #m

        # Control y-axis with left/right arrows
        if self.keys_pressed.get('right', False):
            relative_change[1] += self.delta_pos #m
        if self.keys_pressed.get('left', False):
            relative_change[1] -= self.delta_pos #m
            
        # z-axis with up/down arrows
        if self.keys_pressed.get('up', False):
            relative_change[2] += self.delta_pos #m
        if self.keys_pressed.get('down', False):
            relative_change[2] -= self.delta_pos #m
        
        # Control gripper with a/d keys
        if self.keys_pressed.get('d', False):
            gripper_change += change_gripper_by #rad
        if self.keys_pressed.get('a', False):
            gripper_change -= change_gripper_by #rad

        # Find delta_q from jacobian
        J = self.robot.solve_jacobian(self.current_joints[:4], unit="radians")
        J_inv = np.linalg.pinv(J[:3, :])
        delta_q = np.clip(J_inv @ relative_change, -max_joint_move, max_joint_move) 
        
        #make sure that the gripper cannot be commanded to overactuate
        if self.current_joints[4] + gripper_change > np.pi/2 or self.current_joints[4] + gripper_change < -np.pi/2:
            gripper_change = 0

        total_joint_change = np.append(delta_q, gripper_change)

        # Update current position by adding current and delta
        self.current_position = [x + d_x for x, d_x in zip(self.current_position, relative_change)]
        self.current_joints = [q + d_q for q, d_q in zip(self.current_joints, total_joint_change)]
        
        # Create trajectory message if the changes are nonzero
        #this if statement will pass if there are any nonzero elements in total_joint_change
        if len(total_joint_change[np.nonzero(total_joint_change)]):
            msg = JointTrajectory()
            now = self.get_clock().now()
            msg.header.stamp = now.to_msg()
        
            point = JointTrajectoryPoint()
            point.positions = self.current_joints 
            msg.points = [point]
            #print("Message for trajecory publisher: ", msg)
            self._publisher.publish(msg)
            print(f"Current Position: {np.round(self.current_position, 4)} [m]\nCurrent Joint Angles: {np.round(self.current_joints, 4)}", end="\r\r")
        
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
