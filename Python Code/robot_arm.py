import numpy as np 
from numpy import sin, cos
from warnings import warn
from scipy.optimize import minimize
from time import time
import sympy as sym
from numpy.linalg import inv 
import matplotlib.pyplot as plt
import subprocess as sub
import pandas as pd 
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

class Edubot():
    def __init__(self):
        #Joint bounds
        self.q0Bounds = (-90, 100) #Just a guess
        self.q1Bounds = (-60, 90) #Just a guess
        self.q2Bounds = (-90, 60) #Just a guess
        self.q3Bounds = (-100, 90) #Just a guess
        

        #Link Lengths (DONE)
        #THESE ARE THE MEASURED JOINTS
        self.l0 = 0.045 #ground link to q0
        self.l1 = 0.025 #q0 to q1
        self.l2 = 0.095 #q1 to q2
        self.l3 = 0.105 #q2 to q3
        self.l4 = 0.075 #q3 to end-effector
        self.j2_offset = 0.017 

        #Get jacobian symbolic expression at instantiation of class so it doesnt have to be repeatedly calculated
        self.J = self.get_jacobian()


    #Forward Kinematics(Intermediate Frames based on diagram given in assignment. All joints rotate about +x in the joint frame)
    def forward_kinematics(self, jointAngles, units="radians"):
        if units.lower() == "degrees":
            q0 = jointAngles[0] * np.pi / 180; q1 = jointAngles[1]* np.pi / 180; q2 = jointAngles[2]* np.pi / 180; q3 = jointAngles[3]* np.pi / 180
        elif units.lower() == "radians": 
            q0 = jointAngles[0]; q1 = jointAngles[1]; q2 = jointAngles[2]; q3 = jointAngles[3]
        else:
            warn("Incorrect unit input for forward kinematics function!")
        
        #Create Transformation Matrices
        #1: Base to J0: up by l0 in +z and rotated by q0 in +z
        t0 = np.array([[cos(q0), -sin(q0), 0, 0], 
                       [sin(q0), cos(q0), 0, 0],
                       [0, 0, 1, self.l0],
                       [0, 0, 0, 1]])
        #print("t0 itself\n", t0, "\n")
        
        #2: J0 to J1: up by l1 in +z and then rotate q1 deg about +y (ensure J1 rotates about +y)
        t1 = np.array([
            [cos(q1), 0, sin(q1), -self.j2_offset],
            [0, 1, 0, 0],
            [-sin(q1), 0, cos(q1), self.l1],
            [0, 0, 0, 1],
        ])
        #print("t1 itself\n", np.round(t1, 2), "\n")
        #print("After t1\n", np.round(t0 @ t1, 2), "\n")
        
        # J1 to J2: First rotate the frame such that it matches the simulation and translate l2 in z. Then, rotate frame q2 in y 

        t2_transformation = np.array([
            [0, 0, -1, 0],
            [0, -1, 0, 0],
            [-1, 0, 0, self.l2],
            [0, 0, 0, 1]]
            )
        
        t2_actuation = np.array([
            [cos(q2), 0, sin(q2), 0],
            [0, 1, 0, 0],
            [-sin(q2), 0, cos(q2), 0],
            [0, 0, 0, 1],
        ])
        t2 = t2_transformation @ t2_actuation #combine the two
        #print("t2 itself\n", np.round(t2, 3), "\n")
        #print("After t2_inter\n", np.round(t0 @ t1 @ t2_transformation, 2), "\n")
        #print("After t2\n", np.round(t0 @ t1 @ t2, 2), "\n")
        
        # J2 to J3: First rotate 180deg about z to match robot frame and translate l3 in y. Then rotate frame q3 in y
        t3_transformation = np.array([
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, self.l3],
            [0, 0, 0, 1]]
            )
        t3_actuation = np.array([
            [cos(q3), 0, sin(q3), 0],
            [0, 1, 0, 0],
            [-sin(q3), 0, cos(q3), 0],
            [0, 0, 0, 1],
        ])

        t3 = t3_transformation @ t3_actuation
        #print("t3 itself\n", np.round(t3, 2), "\n")
        #print("After t3\n", np.round(t0 @ t1 @ t2 @ t3, 2), "\n")
        
        # J3 to EE: no rotation, but translate l4 in z
        t4 = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, self.l4],
            [0, 0, 0, 1]
        ])
        #print("t4 itself\n", np.round(t4, 2), "\n")
        
        #Calculate Final Position
        final_transformation = t0 @ t1 @ t2 @ t3 @ t4
        #print("final_transformation\n", np.round(final_transformation, 2))
        final_pos = final_transformation[:3, 3]
        if units == "degrees": 
            final_elevation = q1 - q2 + q3
        else:
            final_elevation = q1 - q2 + q3

        return final_pos, final_elevation

    #Inverse Kinematics(Orientation Agnostic)
    def inverse_kinematics_optimization(self, target_position, initial_guess=None):
        x_target, y_target, z_target = target_position

        #First step, solve for q0 analytically
        #Check for Singlarity(For q0, this happens at x = y = 0)
        if x_target == 0 and y_target == 0:
            warn("The robot is in singularity!")
            q0 = 0
        else: 
            #due to the geometry of the robot, q0 is just the arctan. 
            #also to note, since joints rotate about +z but starts in -x, a positive rotation means the robot goes into negative y and negative x
            print("target x: ", x_target, "\ntarget y: ", y_target)
            q0 = np.arctan2(-y_target, -x_target)

        q0 = np.clip(q0, (self.q0Bounds[0] * np.pi / 180), (self.q0Bounds[1]* np.pi / 180))
        
        #Since there will be multiple solutions, solve numerically for the rest of the angles
        #Method: use scipy's minimize to optimize for least squares errors
        if initial_guess == None: 
            initial_guess = [q0, 0, 0, 0]
        
        bounds = self.get_bounds("radians")
        
        #Objective function.
        def objective(q):
            actual_position, _ = self.forward_kinematics(q, "radians")
            error = np.sum((actual_position - target_position)**2) #MSE 
            return error
        
        #Regularized objective in attempt to keep joint angles small if possible
        def regularized_objective(q):
            position_error = objective(q)
            joint_regularization = 0.0005 * np.sum(q**2)
            return position_error + joint_regularization
        
        # Run the optimization
        result = minimize(
            objective,
            initial_guess,
            bounds=bounds,
            method='L-BFGS-B',
            tol=1e-10,
            options={'maxiter': 1000}
        )
        
        # Check if optimization was successful
        if not result.success:
            print(f"Warning: Optimization did not converge. Message: {result.message}")
        
        # Compute final error
        final_pos = self.forward_kinematics(result.x, "radians")
        error = np.linalg.norm(final_pos - target_position)
        print(f"Final position error: {error:.6f}")
        
        return result.x, error
    
    def inverse_kinematics_newton_raphson(self, target_position, plotting=False, lr=1, tol=1e-6, max_iter=1000, initial_guess = None, printMessages=False): 
        
        """
        This function outputs the inverse kinematics found using the newton-raphson method. 
        INPUTS: target_position[m]:np.array, 
        - Optional: tolerance, max iterations, initial guess for joint angles
        OUTPUTS: joint_position[rad]:np.array, final_error[m]
        """
        
        x_target, y_target, z_target = target_position

        #Pull bounds to enforce
        joint_bounds = self.get_bounds("radians")

        #First step, solve for q0 analytically
        #Check for Singlarity(For q0, this happens at x = y = 0)
        if x_target == 0 and y_target == 0:
            warn("The robot is in singularity!")
            q0 = 0
        else: 
            #due to the geometry of the robot, q0 is just the arctan. 
            #also to note, since joints rotate about +z but starts in -x, a positive rotation means the robot goes into negative y and negative x
            q0 = np.arctan2(-y_target, -x_target)
            #if q0 is within the bounds, use q0
            if q0 > joint_bounds[0][0] and q0 < joint_bounds[0][1]:
                #q0 is within bounds. Keep it pushin
                joint_status = "within"
                q0 = q0 
            elif q0 < joint_bounds[0][0]: 
                #Optimal rotation is less than bottom bound. Shift desired joint by 180deg back into possible workspace
                joint_status = "below"
                q0 += np.pi 
            elif q0 > joint_bounds[0][1]:
                #Optimal rotation is higher than top bound. Shift desired joint by -180deg back into possible workspace
                joint_status = "above"
                q0 -= np.pi
            
        q0 = np.clip(q0, (self.q0Bounds[0] * np.pi / 180), (self.q0Bounds[1]* np.pi / 180))
        
        #Now form the initial guess of position
        if initial_guess is None: 
            q = np.array([q0, 0, 0, 0])
        else: 
            q = np.array([q0, initial_guess[1], initial_guess[2], initial_guess[3]])

        #Create the variables for plotting: 
        iters = np.arange(max_iter)
        qs = np.zeros((4, max_iter))
        errors = np.zeros(max_iter)

        #Now iteratively solve for root of equation
        for i in range(max_iter):
            #Step 1: Calculate the current robot arm position and error between current and desired position
            current_position, current_elevation = self.forward_kinematics(q, "radians")
            delta_x = (target_position - current_position)
            
            
            # Intermediate: Check if the average difference of the last five errors is less than tol(to go faster on unsolvable ones)
            if i >= 10:
                avg_diff = np.mean(np.abs(np.diff(errors[i-10:i+1])))
                if avg_diff < tol:
                    break

        
            error = np.linalg.norm(delta_x)
            errors[i] = error

            if error < tol:
                found_within_tolerance = True
                break
            #Step 2: Calculate the jacobian and inverse jacobian 
            J = self.solve_jacobian(q, "radians")
            J_inv = np.linalg.pinv(J[:3, :])

            #Step 3: Calculate the change in q required: 
            delta_q = J_inv @ (lr * delta_x)
            q[1:] = q[1:] + delta_q[1:]

            #Now enforce joint limits and save joint positions for plotting
            for j in range(len(q)):
                q[j] = np.clip(q[j], joint_bounds[j][0], joint_bounds[j][1])
                qs[j, i] = q[j]
            
            #Warn in case solution is not found
            if i == max_iter-1:
                warn("Inverse_Kinematics_Newton_Raphson: Maximum Iterations Reached Without Finding Joint Angles!")

        if printMessages: 
            print(f"Requested joint is {joint_status} specified bounds")
            print(f"Desired q0 after adjustment: {q0}")
            print(f"target_position = {target_position}")
            print(f"q = {q}")
            print(f"current_position = {current_position}")
            print(f"delta_x = {delta_x}")

        if plotting: 
            plt.figure()
            nonzero_mask = np.nonzero(errors)
            plt.plot(iters[nonzero_mask], errors[nonzero_mask])
            plt.xlabel('Iteration')
            plt.ylabel('Error[m]')
            plt.title('Positional Error vs Iteration')

            plt.figure()
            plt.plot(iters[nonzero_mask], qs[0, nonzero_mask].flatten(), label="q0")
            plt.plot(iters[nonzero_mask], qs[1, nonzero_mask].flatten(), label="q1")
            plt.plot(iters[nonzero_mask], qs[2, nonzero_mask].flatten(), label="q2")
            plt.plot(iters[nonzero_mask], qs[3, nonzero_mask].flatten(), label="q3")
            plt.legend()
            plt.xlabel('Iteration')
            plt.ylabel('Joint Angle [rad]')
            plt.title('Joint Angles vs Iteration')
            
            plt.show()
        
        return q, error

    def inverse_kinematics_analytical(self, target_position, target_elevation):
        #This is mostly from Feitong's analytical kinematic function
        x_target, y_target, z_target = target_position

        #Pull bounds to enforce
        joint_bounds = self.get_bounds("radians")

        if x_target == 0 and y_target == 0:
            warn("The robot is in singularity!")
            q0 = 0
        else: 
            #due to the geometry of the robot, q0 is just the arctan. 
            #also to note, since joints rotate about +z but starts in -x, a positive rotation means the robot goes into negative y and negative x
            q0 = np.arctan2(-y_target, -x_target)
            #if q0 is within the bounds, use q0
            if q0 > joint_bounds[0][0] and q0 < joint_bounds[0][1]:
                #q0 is within bounds. Keep it pushin
                joint_status = "within"
                q0 = q0 
            elif q0 < joint_bounds[0][0]: 
                #Optimal rotation is less than bottom bound. Shift desired joint by 180deg back into possible workspace
                joint_status = "below"
                q0 += np.pi 
            elif q0 > joint_bounds[0][1]:
                #Optimal rotation is higher than top bound. Shift desired joint by -180deg back into possible workspace
                joint_status = "above"
                q0 -= np.pi

        x_target, y_target, z_target = target_position
        h = z_target - self.l0 - self.l1
        l1 = self.l2
        l2 = self.l3
        l3 = self.l4

        alpha = target_elevation
        
        length = np.sqrt(x_target**2 + y_target**2)
        L = length - l3*np.cos(alpha)
        H = h - l3*np.sin(alpha)

        # solve for q2
        s2 = (L**2 + H**2 - l1**2 - l2**2) / (-2*l1*l2)

        c2 = np.sqrt(1 - s2**2)
        q2 = np.arctan2(s2, c2)

        # solve for q1
        k1 = l1 - l2*s2
        k2 = l2*c2
        w = np.arctan2(k2, k1)
        q1 = w - np.arctan2(L, H)

        # solve for q3
        q3 = alpha - q1 + q2

        q = np.array([q0, q1, q2, q3])
        q_preclipping = q
        #Now enforce joint limits and save joint positions for plotting
        for j in range(len(q)):
            q[j] = np.clip(q[j], joint_bounds[j][0], joint_bounds[j][1])


        if np.all(q == q_preclipping):
            fully_solved = True
        else:
            fully_solved = False

        return q, fully_solved
    
    def get_bounds(self, units="degrees"):
        if units.lower() == "degrees":
            return [self.q0Bounds, self.q1Bounds, self.q2Bounds, self.q3Bounds]
        elif units.lower() == "radians":
            return [(self.q0Bounds[0] * np.pi / 180, self.q0Bounds[1] * np.pi / 180), (self.q1Bounds[0] * np.pi / 180, self.q1Bounds[1] * np.pi / 180), (self.q2Bounds[0] * np.pi / 180, self.q2Bounds[1] * np.pi / 180), (self.q3Bounds[0] * np.pi / 180, self.q3Bounds[1] * np.pi / 180)]
        else: 
            warn("There was an incorrect unit description inside the get_bounds() function!")
    
    def get_jacobian(self, print_final_transformation=False):
        q0 = sym.Symbol("q0"); q1 = sym.Symbol("q1"); q2 = sym.Symbol("q2"); q3 = sym.Symbol("q3")

        #Copying transformation from forward kinematics function

        #1: Base to J0: up by l0 in +z and rotated by q0 in +z
        t0 = np.array([[sym.cos(q0), -sym.sin(q0), 0, 0], 
                       [sym.sin(q0), sym.cos(q0), 0, 0],
                       [0, 0, 1, self.l0],
                       [0, 0, 0, 1]])
        
        #2: J0 to J1: up by l1 in +z and then rotate q1 deg about +y (ensure J1 rotates about +y)
        t1 = np.array([
            [sym.cos(q1), 0, sym.sin(q1), -self.j2_offset],
            [0, 1, 0, 0],
            [-sym.sin(q1), 0, sym.cos(q1), self.l1],
            [0, 0, 0, 1],
        ])
        
        # J1 to J2: First rotate the frame such that it matches the simulation and translate l2 in z. Then, rotate frame q2 in y 

        t2_transformation = np.array([
            [0, 0, -1, 0],
            [0, -1, 0, 0],
            [-1, 0, 0, self.l2],
            [0, 0, 0, 1]]
            )
        
        t2_actuation = np.array([
            [sym.cos(q2), 0, sym.sin(q2), 0],
            [0, 1, 0, 0],
            [-sym.sin(q2), 0, sym.cos(q2), 0],
            [0, 0, 0, 1],
        ])
        t2 = t2_transformation @ t2_actuation #combine the two
        
        # J2 to J3: First rotate 180deg about z to match robot frame and translate l3 in y. Then rotate frame q3 in y
        t3_transformation = np.array([
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, self.l3],
            [0, 0, 0, 1]]
            )
        t3_actuation = np.array([
            [sym.cos(q3), 0, sym.sin(q3), 0],
            [0, 1, 0, 0],
            [-sym.sin(q3), 0, sym.cos(q3), 0],
            [0, 0, 0, 1],
        ])

        t3 = t3_transformation @ t3_actuation
        
        # J3 to EE: no rotation, but translate l4 in z
        t4 = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, self.l4],
            [0, 0, 0, 1]
        ])
        
        #Calculate Final Position
        final_transformation = t0 @ t1 @ t2 @ t3 @ t4
        if print_final_transformation: 
            print(sym.simplify(final_transformation))

        x = final_transformation[0, 3]
        y = final_transformation[1, 3]
        z = final_transformation[2, 3]

        j_11 = sym.diff(x, q0); j_12 = sym.diff(x, q1); j_13 = sym.diff(x, q2); j_14 = sym.diff(x, q3)
        j_21 = sym.diff(y, q0); j_22 = sym.diff(y, q1); j_23 = sym.diff(y, q2); j_24 = sym.diff(y, q3)
        j_31 = sym.diff(z, q0); j_32 = sym.diff(z, q1); j_33 = sym.diff(z, q2); j_34 = sym.diff(z, q3)
        j_elev1 = 0; j_elev2 = 1; j_elev3 = -1; j_elev4 = 1; #Relation of each joint to the elevation 

        #Note here that the angle about z is fixed depending on desired positon, and since there is no wrist
        #there is no rotation about the final x-axis, so all we care about is elevation

        jacobian = sym.Matrix([[j_11, j_12, j_13, j_14],
                              [j_21, j_22, j_23, j_24],
                              [j_31, j_32, j_33, j_34], 
                              [j_elev1, j_elev2, j_elev3, j_elev4]])
        
        return jacobian
    
    def solve_jacobian(self, q_array, unit="degrees"):
        
        #No matter what, we want radian inputs to the jacobian, so convert if necessary
        if unit.lower() == "degrees":
            #Convert Degree Input to Radians
            q_array = q_array * np.pi / 180
        elif unit.lower() == "radians":
            #change nothing
            pass 
        else: 
            warn("Incorrect Argument for Unit in Solve_Jacobian Function!")

        q0 = sym.Symbol("q0"); q1 = sym.Symbol("q1"); q2 = sym.Symbol("q2"); q3 = sym.Symbol("q3")
        q_sol = {q0: q_array[0], 
                 q1: q_array[1], 
                 q2: q_array[2], 
                 q3: q_array[3]}
        
        solved_jacobian = self.J.subs(q_sol)
        j_array = np.asarray(solved_jacobian)
        return j_array.astype(float)

    def trace_cartesian_trajectory(self, filename, height, dist, generateOutline=False, saveOutput=False, previewTrajectory=False):
        """ 
        This function has two possible options: 
        1) generateOutline=False: Take in a pre-generated csv of outlined trajectory and convert it to cartesian trajectory list for the robot
        2) generateOutline=True: Take in an image file, generate an outline csv file for it using outliner.cpp, and then convert it to cartesian trajectory
        """

        #If generating the outline first, then use the C++ code to generate the output and save it 
        if generateOutline: 
            prefix = filename.split(".")[0]
            outputName = prefix + ".csv"
            result = sub.run(["./Henrys_Code/outliner", filename, outputName], capture_output=True, text=True)
            if result.returncode == 0:
                print("Success! Output:", result.stdout)
            else: 
                print("Error running the executable:", result.stderr)
            filename = outputName
        
        outline = pd.read_csv(filename)

        csv_x = outline.x
        csv_y = outline.y

        #Now Scale To Some Reasonable Height To Define the Geometry
        #define the geometry
        min_height = 0.1 #m
        cosnt_x = -dist
        width = height * (np.max(csv_x) - np.min(csv_x))/(np.max(csv_y) - np.min(csv_y))

        #scale the actual trajectory
        normalized_y = (csv_y - np.min(csv_y)) / np.max(csv_y - np.min(csv_y))
        normalized_x = (csv_x - np.min(csv_x)) / np.max(csv_x - np.min(csv_x)) - 0.5 #normalize and center
        print(np.min(normalized_x), np.max(normalized_x))

        #Since we are plotting the actual trajectory in constant x but varying y and z, map x->y and y->z
        actual_x = np.ones(len(normalized_x)) * cosnt_x
        actual_y = normalized_x * width
        actual_z = normalized_y * height; actual_z = (np.max(actual_z) - actual_z) + min_height #flip the z-axis bcs flame is upside down

        cartesian_trajectory = np.array([actual_x, actual_y, actual_z])

        #Save the output to a csv file if asked to
        if saveOutput: 
            traj_dict = {"x" : actual_x, "y" : actual_y, "z" : actual_z}
            traj_df = pd.DataFrame(traj_dict)
            outputName = filename.split(".")[0] + "_cartesian_trajectory.csv"
            traj_df.to_csv(outputName)
        
        #plot trajectory if you're asked for it
        if previewTrajectory:
            plt.figure()
            ax = plt.axes(projection ='3d')
            ax.scatter(actual_x, actual_y, actual_z)
            ax.set_title('3D Cartesian Trajectory of Robot')
            ax.set_xlabel("x[m]")
            ax.set_ylabel("y[m]")

            plt.figure()
            plt.plot(actual_y, actual_z)
            plt.xlabel("Robot y-axis")
            plt.ylabel("Robot z-axis")
            plt.title("2D Projection of Robot Trajectory onto yz plane")
            plt.show()

        return cartesian_trajectory
    
    def cartesian_to_joint_trajectory(self, fromFile, cartesianArray, filename, level_ee=False, saveOutput=False, checkForwardKinematics=False): 
        
        #load in the x, y, and z 
        if fromFile:
            cartesian_points = pd.read_csv(filename)
            x = cartesian_points.x
            y = cartesian_points.y
            z = cartesian_points.z
            cartesianArray = np.array([x, y, z]).T         
        
        q_array = np.zeros((len(x), 4))

        #iterate through the points and find inverse kinematics at each
        guess = np.array([0, 0, 0, 0]) #Start with no guess but update based on where robot is

        #For Debugging: Check what points get clipped
        clipped = 0

        
        for idx, point in enumerate(cartesianArray):
            if level_ee:
                #try to keep the end-effector level using the analytical function
                q_i, solved = self.inverse_kinematics_analytical(point, 0)
                reconstructed_point, _ = self.forward_kinematics(q_i)
                error = np.linalg.norm(point - reconstructed_point)
                if error > 0.001:
                    print(f"Point:  {point} could not be reached with output {q_i} and resulted in reconstructed position {reconstructed_point}")
                    break

                if not solved:
                    clipped +=1 
                q_array[idx, :] = q_i
        
            else:
                #use the newton-raphson solver
                [q_i, error_i] = self.inverse_kinematics_newton_raphson(point, initial_guess=guess)
                if error_i >  1e-5:
                    print(f"Point:  {point} could not be reached using initial guess: {guess} and output {q_i}")
                    break
                
                guess = q_i #update guess for faster solutions

                #update the q_array object to save info
                #print(f"Replace {q_array[idx, :]} with {q_i}")
                q_array[idx, :] = q_i

        print(f"Total of {clipped} points clipped(aka were out of robot bounds)")
        #after it iterates through all these points, save joint trajectory to a .csv file 
        if saveOutput:
            q_traj_dict = {"q0":q_array[:, 0], "q1":q_array[:, 1], "q2":q_array[:, 2], "q3":q_array[:, 3]}
            q_traj_df = pd.DataFrame(q_traj_dict)
            outputName = filename.split(".")[0]; outputName = outputName.strip("cartesian_trajectory") + "level_joint_trajectory.csv"
            q_traj_df.to_csv(outputName)

        #If needed, check trajectory by reconstructing with forward kinematics
        if checkForwardKinematics: 
            point_array = np.zeros((np.size(q_array, 0), 3))
            for idx, q_i in enumerate(q_array):
                point_i, _ = self.forward_kinematics(q_i)
                point_array[idx, :] = point_i
            reconstruction_dict = {"x": point_array[:, 0],"y": point_array[:, 1],"z": point_array[:, 2]}
            reconstructon_df = pd.DataFrame(reconstruction_dict)
            reconstructon_df.to_csv("flame_reconstruction.csv")

            plt.figure()
            ax = plt.axes(projection ='3d')
            ax.scatter(point_array[:,0],point_array[:,1], point_array[:,2], c="green")
            ax.set_title("Reconstructed Cartesian Trajectory")
            plt.show()

        return q_array
    
if __name__=="__main__":
    t_start = time()
    robot = Edubot()
    t_end = time()
    print(f"Time to instantiate: {t_end - t_start}")
    
    #Define what we're checking:
    fk = False
    ik = False
    cartesian_trajectory = False
    joint_trajectory = False
    test_reconstruction = False
    get_symbolic_jacobian = True

    #Now Do Whatever Tests are Specified
    if fk: 
        # CHECKING THE FORWARD KINEMATICS TO MAKE SURE THEY MAKE SENSE
        pos = robot.forward_kinematics(np.array([np.pi/4, 0, 0, 0]), "radians")
        print(pos)

    if ik:
        # CHECKING THE INVERSE KINEMATICS FUNCTION TO MAKE SURE IT WORKS
        #point1 = np.array([-0.1, 0.1, 0.1])
        point1 = np.array([-0.1, 0.1 ,0.03])
        [q_array_1, errors1] = robot.inverse_kinematics_newton_raphson(point1, plotting=False)
        reconstruction1 = robot.forward_kinematics(q_array_1, "radians")
        print(f"WITH NO ELEV CONSTRAINTS:\nq's to achieve {point1} from jacobian algorithm: {q_array_1} with error: {errors1}.\nReconstructed position: {reconstruction1}\n")  

        #Now try same thing with Analytical IK, specifying the angle
        desired_elev = 0
        q_array_1, _ = robot.inverse_kinematics_analytical(point1, desired_elev)
        reconstruction1 = robot.forward_kinematics(q_array_1, "radians")
        print(f"USING ANALYTICAL SOLVER:\nq's to achieve {point1} from jacobian algorithm: {q_array_1} with error: {0}.\nReconstructed position: {reconstruction1}")  
        
    if cartesian_trajectory:
        # CHECKING THE TRACE_CARTESIAN_TRAJECTORY POINT WITH generateOutline ENABLED AND DISABLED
        robot.trace_cartesian_trajectory("Henrys_Code/tu_flame.csv", 0.15, 0.2, generateOutline=False, saveOutput=True, previewTrajectory=True)
        
    if joint_trajectory:
        #CHECKING THE CARTESIAN_TO_JOINT_TRAJECTORY() FUNCTION
        robot.cartesian_to_joint_trajectory(True, None, "Henrys_Code/tu_flame_cartesian_trajectory.csv", level_ee=False, saveOutput=True, checkForwardKinematics=True)

    if test_reconstruction: 
        actual_data = pd.read_csv("Henrys_Code/tu_flame_cartesian_trajectory.csv")
        recon_data = pd.read_csv("Henrys_Code/flame_reconstruction.csv")

        actual_points = np.array([actual_data.x, actual_data.y, actual_data.z]).T
        recon_points = np.array([recon_data.x, recon_data.y, recon_data.z]).T

        point_idx = np.arange(np.size(actual_points, 0))
        diffs = np.zeros(np.size(actual_points, 0))
        for idx, actual_point in enumerate(actual_points):
            recon_point = recon_points[idx, :]
            diff = actual_point - recon_point
            total_diff = np.linalg.norm(diff)
            diffs[idx] = total_diff

        plt.figure()
        plt.plot(point_idx, diffs)

        plt.figure()
        ax = ax = plt.axes(projection ='3d')
        ax.scatter(actual_points[:, 0], actual_points[:, 1], actual_points[:, 2])

        plt.figure()
        ax = ax = plt.axes(projection ='3d')
        ax.scatter(recon_points[:, 0], recon_points[:, 1], recon_points[:, 2])

        plt.show()
            
    if get_symbolic_jacobian: 
        robot.get_jacobian(True)


