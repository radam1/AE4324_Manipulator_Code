#include "robot_arm.hpp"
#include <cmath>
#include <iostream>

using namespace Eigen;
using namespace std;

double deg2rad(float angle_degrees){
    return angle_degrees * M_PI / 180.0; // make sure the 180 is 180.0 for floating point math 
}

/*
This is the constructor. It allows you to 
*/ 
Edubot::Edubot() {
    q0Bounds = std::make_pair(-90, 90); 
    q1Bounds = std::make_pair(-90, 90); 
    q2Bounds = std::make_pair(-90, 90); 
    q3Bounds = std::make_pair(-90, 90); 

    l0 = 0.045; 
    l1 = 0.025; 
    l2 = 0.095;  
    l3 = 0.105;  
    l4 = 0.075;  
    j2_offset = 0.006; 
}; 

void Edubot::warn(const std::string& message) {
    std::cerr << "WARNING: " << message << std::endl;
    };

void Edubot::debug_print(const std::string& message) {
    std::cerr << "DEBUG: " << message << std::endl;
    };

// Forward kinematics implementation
Vector3d Edubot::forward_kinematics(const Vector4d& jointAngles, const std::string& units) {
    double q0, q1, q2, q3;
    
    if (units == "degrees") {
        q0 = deg2rad(jointAngles[0]);
        q1 = deg2rad(jointAngles[1]);
        q2 = deg2rad(jointAngles[2]);
        q3 = deg2rad(jointAngles[3]);
    } 
    else if (units == "radians") {
        q0 = jointAngles[0];
        q1 = jointAngles[1];
        q2 = jointAngles[2];
        q3 = jointAngles[3];
    } 
    else {
        warn("Incorrect unit input for forward kinematics function!");
        q0 = jointAngles[0];
        q1 = jointAngles[1];
        q2 = jointAngles[2];
        q3 = jointAngles[3];
    }
    
    // Create transformation matrices
    // 1: Base to J0: up by l0 in +z and rotated by q0 in +z
    Matrix4d t0; 
    t0 << cos(q0), -sin(q0), 0, 0, 
          sin(q0), cos(q0), 0, 0,
          0, 0, 1, l0,
          0, 0, 0, 1; 

    // J0 to J1
    Matrix4d t1; 
    t1 << cos(q1), 0, sin(q1), j2_offset,
        0, 1, 0, 0,
        -sin(q1), 0, cos(q1), l1,
        0, 0, 0, 1;

    Matrix4d t2_transformation; 
    Matrix4d t2_actuation; 
    Matrix4d t2; 

    // j1 to j2
    t2_transformation << 0, 0, -1, 0,
                        0, -1, 0, 0,
                        -1, 0, 0, l2,
                        0, 0, 0, 1;

    t2_actuation << cos(q2), 0, sin(q2), 0,
                    0, 1, 0, 0,
                    -sin(q2), 0, cos(q2), 0,
                    0, 0, 0, 1;

    t2 = t2_transformation * t2_actuation; 

    // j2 to j3
    Matrix4d t3; 
    Matrix4d t3_transformation; 
    Matrix4d t3_actuation; 

    t3_transformation << -1, 0, 0, 0,
    0, -1, 0, 0,
    0, 0, 1, l3,
    0, 0, 0, 1; 

    t3_actuation << cos(q3), 0, sin(q3), 0,
                    0, 1, 0, 0,
                    -sin(q3), 0, cos(q3), 0,
                    0, 0, 0, 1;

    t3 = t3_transformation * t3_actuation; 

    // j3 to ee 
    Matrix4d t4; 
    t4 << 1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, l4,
          0, 0, 0, 1;

    // Calculate final position
    Matrix4d final_transformation = t0 * t1 * t2 * t3 * t4;
    Vector3d final_pos = final_transformation.block<3,1>(0,3);
    
    return final_pos;
}

MatrixXd Edubot::solve_jacobian(Vector4d& q_array, const std::string& unit = "radians") { 
    double q0, q1, q2, q3;

    // Handle the degree/radian conversions if necessary
    if (unit == "degrees") {
        q0 = deg2rad(q_array[0]);
        q1 = deg2rad(q_array[1]);
        q2 = deg2rad(q_array[2]);
        q3 = deg2rad(q_array[3]);
    } else if (unit == "radians") {
        q0 = q_array[0];
        q1 = q_array[1];
        q2 = q_array[2];
        q3 = q_array[3];
    } else {
        warn("Incorrect Argument for Unit in Solve_Jacobian Function!");
        q0 = q_array[0];
        q1 = q_array[1];
        q2 = q_array[2];
        q3 = q_array[3];
    }

    // 3x4 Jacobian, so only pertaining to positon (xyz) and joint angles (j0 to j3)
    double j11, j12, j13, j14; 
    double j21, j22, j23, j24;  
    double j31, j32, j33, j34;  

    // used symbolic jacobian found in python to hard-code this jacobian 
    j11 = 0.075*(1.0*sin(q0)*sin(q1)*sin(q2) + 1.0*sin(q0)*cos(q1)*cos(q2))*cos(q3) - 0.075*(1.0*sin(q0)*sin(q1)*cos(q2) - 1.0*sin(q0)*sin(q2)*cos(q1))*sin(q3) + 0.105*sin(q0)*sin(q1)*sin(q2) - 0.095*sin(q0)*sin(q1) + 0.105*sin(q0)*cos(q1)*cos(q2) + 0.006*sin(q0); 
    j12 = -0.075*(-1.0*sin(q1)*sin(q2)*cos(q0) - 1.0*cos(q0)*cos(q1)*cos(q2))*sin(q3) + 0.075*(1.0*sin(q1)*cos(q0)*cos(q2) - 1.0*sin(q2)*cos(q0)*cos(q1))*cos(q3) + 0.105*sin(q1)*cos(q0)*cos(q2) - 0.105*sin(q2)*cos(q0)*cos(q1) + 0.095*cos(q0)*cos(q1);
    j13 = -0.075*(1.0*sin(q1)*sin(q2)*cos(q0) + 1.0*cos(q0)*cos(q1)*cos(q2))*sin(q3) + 0.075*(-1.0*sin(q1)*cos(q0)*cos(q2) + 1.0*sin(q2)*cos(q0)*cos(q1))*cos(q3) - 0.105*sin(q1)*cos(q0)*cos(q2) + 0.105*sin(q2)*cos(q0)*cos(q1);
    j14 = -0.075*(-1.0*sin(q1)*sin(q2)*cos(q0) - 1.0*cos(q0)*cos(q1)*cos(q2))*sin(q3) - 0.075*(-1.0*sin(q1)*cos(q0)*cos(q2) + 1.0*sin(q2)*cos(q0)*cos(q1))*cos(q3);
    j21 =  0.075*(-1.0*sin(q1)*sin(q2)*cos(q0) - 1.0*cos(q0)*cos(q1)*cos(q2))*cos(q3) - 0.075*(-1.0*sin(q1)*cos(q0)*cos(q2) + 1.0*sin(q2)*cos(q0)*cos(q1))*sin(q3) - 0.105*sin(q1)*sin(q2)*cos(q0) + 0.095*sin(q1)*cos(q0) - 0.105*cos(q0)*cos(q1)*cos(q2) - 0.006*cos(q0);
    j22 = -0.075*(-1.0*sin(q0)*sin(q1)*sin(q2) - 1.0*sin(q0)*cos(q1)*cos(q2))*sin(q3) + 0.075*(1.0*sin(q0)*sin(q1)*cos(q2) - 1.0*sin(q0)*sin(q2)*cos(q1))*cos(q3) + 0.105*sin(q0)*sin(q1)*cos(q2) - 0.105*sin(q0)*sin(q2)*cos(q1) + 0.095*sin(q0)*cos(q1);
    j23 =  -0.075*(1.0*sin(q0)*sin(q1)*sin(q2) + 1.0*sin(q0)*cos(q1)*cos(q2))*sin(q3) + 0.075*(-1.0*sin(q0)*sin(q1)*cos(q2) + 1.0*sin(q0)*sin(q2)*cos(q1))*cos(q3) - 0.105*sin(q0)*sin(q1)*cos(q2) + 0.105*sin(q0)*sin(q2)*cos(q1);
    j24 = -0.075*(-1.0*sin(q0)*sin(q1)*sin(q2) - 1.0*sin(q0)*cos(q1)*cos(q2))*sin(q3) - 0.075*(-1.0*sin(q0)*sin(q1)*cos(q2) + 1.0*sin(q0)*sin(q2)*cos(q1))*cos(q3);
    j31 = 0.0;
    j32 =  0.075*(1.0*sin(q1)*sin(q2) + 1.0*cos(q1)*cos(q2))*cos(q3) - 0.075*(1.0*sin(q1)*cos(q2) - 1.0*sin(q2)*cos(q1))*sin(q3) + 0.105*sin(q1)*sin(q2) - 0.095*sin(q1) + 0.105*cos(q1)*cos(q2);
    j33 = 0.075*(-1.0*sin(q1)*sin(q2) - 1.0*cos(q1)*cos(q2))*cos(q3) - 0.075*(-1.0*sin(q1)*cos(q2) + 1.0*sin(q2)*cos(q1))*sin(q3) - 0.105*sin(q1)*sin(q2) - 0.105*cos(q1)*cos(q2);
    j34 = -0.075*(-1.0*sin(q1)*sin(q2) - 1.0*cos(q1)*cos(q2))*cos(q3) - 0.075*(1.0*sin(q1)*cos(q2) - 1.0*sin(q2)*cos(q1))*sin(q3);

    MatrixXd jacobian(3, 4); 

    jacobian << j11, j12, j13, j14, 
                j21, j22, j23, j24,
                j31, j32, j33, j34; 
    
    return jacobian; 
}

std::pair<Eigen::Vector4d, double> Edubot::inverse_kineamtics_newton_raphson(const Vector3d desired_position, 
                                                                             double tol, 
                                                                             int max_iter, 
                                                                             const Eigen::Vector4d* initial_guess){
    
    // extract positions and joint bounds 
    double target_x = desired_position[0]; 
    double target_y = desired_position[0]; 
    double target_z = desired_position[0];

    auto joint_bounds = get_bounds("radians"); 

    //First, get the q0 angle since that is constrained by the 
    double q0; 
    if (target_x == 0 && target_y == 0){
        warn("ROBOT IS IN SINGULARITY! Theoretically infinite starting positions");
        q0 = 0; 
    } else {
        q0 = atan2(-target_y, -target_x); 
        // clipping the bounds and flipping 180deg if necessary since this will give robot best chance at the point
        if (q0 > joint_bounds[0].first && q0 < joint_bounds[0].second){
            //Joint is within bounds. Nice! 
            std::cout << "Desired q0 is within bounds. No adjustment needed" << std::endl;

        }else if (q0 < joint_bounds[0].first){
            //Joint lower than bottom bound. Need to add 180deg to it(pi rads) 
            std::cout << "Desired q0 is below bottom bound. Flipping 180deg." << std::endl;
            q0 += M_PI; 
        }else if (q0 > joint_bounds[0].second){
            std::cout << "Desired q0 is above bound. Flipping 180deg." << std::endl;
            q0 -= M_PI; 
        }
        std::cout << "Desired q0 after adjustment: " << q0 << std::endl;
    }
    Vector4d q; 
    if (initial_guess == nullptr) {
        q = Vector4d(q0, 0, 0, 0);
    } else {
        // quick note here. Since the nullptr is being used, we have to make sure to pass a pointer through the function when we use it. 
        // the * before the initual_guess variable here extracts the value from the pointer
        q = Vector4d(q0, (*initial_guess)[1], (*initial_guess)[2], (*initial_guess)[3]);
    }

    for (int i=1; i<max_iter; i++){
        Vector3d current_position = forward_kinematics(q, "radians"); 
        Vector3d pos_err = desired_position - current_position; 
        double error = pos_err.norm();

        // check if the error is less than the tolerance and if it is break of of the loop
        if (error < tol){
            break; 
        }

        // get the jacobian
        MatrixXd J = solve_jacobian(q, "radians");

        // Eigen has a pseudo-inverse function built in conveniently
        MatrixXd J_inv = J.completeOrthogonalDecomposition().pseudoInverse(); 

        Vector4d delta_q = J_inv * pos_err; 
    }
    }



int main() { 
    Edubot robot = Edubot(); 
    // Test the forward kinematics function 
    Vector4d joint_angles(0, 0, 0, 0); 
    std::string units = "radians"; 
    Vector3d pos = robot.forward_kinematics(joint_angles, units); 
    std::cout << "Point output from " << joint_angles.transpose() << "is the position" << pos.transpose() << std::endl; 

    //Now try finding the joint angles with the inverse kinematics function
    std::pair<Vector4d, double> joint_reconstuction = robot.inverse_kineamtics_newton_raphson(pos); 
    Vector4d reconstructed_joints  = joint_reconstuction.first; 
    double errors = joint_reconstuction.second; 
    std::cout << "Reconstructed Joint For Position" << pos.transpose() << "is the array" << reconstructed_joints.transpose() << std::endl; 
    
    //return 0 for successful run
    return 0; 
}


