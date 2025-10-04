import numpy as np 
from robot_arm import Edubot
from time import time

#This is the answer to part 2.1 for the AE 4324 Assignment
robot = Edubot()
point1 = np.array([0.1, 0.1, 0.1])
point2 = np.array([0.2, 0.1, 0.3])
point3 = np.array([0.0, 0.0, 0.3])
point4 = np.array([0.0, 0.0, 0.07])

#Part 1: Using the optimization algorithm
"""t_start = time()
q_array_1, errors1 = robot.inverse_kinematics_optimization(point1)
reconstruction1 = robot.forward_kinematics(q_array_1)
print(f"q's to achieve {point1} from jacobian algorithm: {q_array_1} with error: {errors1}.\nReconstructed position: {reconstruction1}\n")

q_array_2, errors2 = robot.inverse_kinematics_optimization(point2)
reconstruction2 = robot.forward_kinematics(q_array_2)
print(f"q's to achieve {point2} from jacobian algorithm: {q_array_2} with error: {errors2}.\nReconstructed position: {reconstruction2}\n")

q_array_3, errors3 = robot.inverse_kinematics_optimization(point3)
reconstruction3 = robot.forward_kinematics(q_array_3)
print(f"q's to achieve {point3} from jacobian algorithm: {q_array_3} with error: {errors3}.\nReconstructed position: {reconstruction3}\n")

q_array_4, errors4 = robot.inverse_kinematics_optimization(point4)
reconstruction4 = robot.forward_kinematics(q_array_4)
print(f"q's to achieve {point4} from jacobian algorithm: {q_array_4} with error: {errors4}.\nReconstructed position: {reconstruction4}\n")

t_end = time()

print("Time Elapsed For Optimization: ", t_end - t_start)"""

#Part 2 : Using the Jacobian Algorithm
t_start = time()
q_array_1, errors1 = robot.inverse_kinematics_newton_raphson(point1)
reconstruction1 = robot.forward_kinematics(q_array_1)
print(f"q's to achieve {point1} from jacobian algorithm: {q_array_1} with error: {errors1}.\nReconstructed position: {reconstruction1}\n")

q_array_2, errors2 = robot.inverse_kinematics_newton_raphson(point2)
reconstruction2 = robot.forward_kinematics(q_array_2)
print(f"q's to achieve {point2} from jacobian algorithm: {q_array_2} with error: {errors2}.\nReconstructed position: {reconstruction2}\n")

q_array_3, errors3 = robot.inverse_kinematics_newton_raphson(point3)
reconstruction3 = robot.forward_kinematics(q_array_3)
print(f"q's to achieve {point3} from jacobian algorithm: {q_array_3} with error: {errors3}.\nReconstructed position: {reconstruction3}\n")

q_array_4, errors4 = robot.inverse_kinematics_newton_raphson(point4)
reconstruction4 = robot.forward_kinematics(q_array_4)
print(f"q's to achieve {point4} from jacobian algorithm: {q_array_4} with error: {errors4}.\nReconstructed position: {reconstruction4}\n")

t_end = time()

print("Time Elapsed for Jacobian: ", t_end - t_start)
