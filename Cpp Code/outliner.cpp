#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char** argv){
    // First, check if there are the right number of arguments(cpp file itself, input file, output file)
    if (argc != 3){
        // tell user their implementation is wrong
        std::cout << "Usage: " << argv[0] << "your <input_file> <output_file> is formatted incorrect" << std::endl; 
        // instantly end the program with an error 
        return -1; 
    }

    //Gather image path and read image into a cv:Mat object, then make sure it was loaded correctly
    std::string image_path = argv[1]; 
    cv::Mat image = cv::imread(image_path);

    if (image.empty()){
        std::cout << "Could not open image " << argv[1] << std::endl; 
        return -1; 
    }

    //convert the image to grayscale 
    cv::Mat gray_image; 
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY); 

    // Instead of thresholding, apply canny edge detection
    cv::Mat edges;
    cv::Canny(gray_image, edges, 100, 200);

    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(edges, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Check if any contour was found
    if (contours.empty()) {
        std::cout << "No contours found in the image." << std::endl;
        return -1;
    }

    // Find the largest contour (assumed to be the main object)
    int largestContourIndex = 0;
    double largestContourArea = 0;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area > largestContourArea) {
            largestContourArea = area;
            largestContourIndex = i;
        }
    }

    // get vector of points for the largest contour
    std::vector<cv::Point> outline = contours[largestContourIndex];

    // visualizate outline on top of image
    cv::Mat outlineImage = image.clone();
    cv::drawContours(outlineImage, contours, largestContourIndex, cv::Scalar(255, 0, 0), 2); //draw red rect around shape that was detected
    cv::imwrite("outline_visualization.jpg", outlineImage);

    // Output to CSV file
    std::string csvPath = argv[2];
    std::ofstream csvFile(csvPath);
    
    if (csvFile.is_open()) {
        //header
        csvFile << "x,y" << std::endl;
        
        //points
        for (const auto& point : outline) {
            csvFile << point.x << "," << point.y << std::endl;
        }
        
        csvFile.close();
        std::cout << "Outline coordinates saved to " << csvPath << std::endl;
    } else {
        std::cout << "Failed to open output file: " << csvPath << std::endl;
        return -1;
    }

    std::cout << "Outline extraction completed. Found " << outline.size() << " points." << std::endl;
    return 0;
}
