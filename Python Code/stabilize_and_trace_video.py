import cv2
import numpy as np

# Load video
cap = cv2.VideoCapture("Videos/tu_flame_tracing.mp4")

# For storing tracked points
trajectory = []

# Video writer 
width = 850
height = 900 
marker_placement = (0.925, 0.85)
size = (width, height)
writer = cv2.VideoWriter('traced_flame_stable.mp4',  
                            cv2.VideoWriter_fourcc(*'MJPG'), 
                            20, size) 

#Parameters for debugging
print_mask = True
print_messages = False

def find_base_region(frame):
    #Find the yellow sticker at the bottom and then creates a sub-frame based on that
    roi_lower = frame.shape[0]//2
    roi_upper = 3* frame.shape[0]//4
    roi_left = frame.shape[1]//2
    roi_right = frame.shape[1]
    roi = frame[roi_lower:roi_upper, roi_left:roi_right]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    #Use HSV color range to find yellow sticker(Hue, Saturation, Value)
    #specific yellow range found online
    lower_yellow = np.array([20, 100, 100])  
    upper_yellow = np.array([30, 255, 255])  
    
    #Create mask for yellow objects
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    masked_frame = cv2.bitwise_and(roi, roi, mask=yellow_mask)
    
    #Convert to grayscale(edge detection)
    gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur(probably not necessary but generally good to do)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny Edge detection, same as the trajectory tracing
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    #Detect largest yellow contour and find coodinates of centroid
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)

        if M["m00"] != 0:
            # Use the moments to find the centroid of the sticker
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cx_global = roi_left + cx
            cy_global = roi_lower+cy
            if print_messages:
                print(f"cx = {cx_global}\ncy = {cy_global}")
            width_start = int(cx_global - marker_placement[0] * width )
            width_end =  int(cx_global + (1-marker_placement[0]) * width )
            height_start = int(cy_global - marker_placement[1] * height)
            height_end = int(cy_global + (1-marker_placement[1]) * height)

            # For debugging: Show the mask and edges to make sure its detecting the sticker correctly
            if print_mask:
                cv2.circle(yellow_mask, (cx, cy), 5, (0, 255, 0), -1)
                cv2.circle(frame, (cx_global, cy_global), 5, (0, 255, 0), -1)
                cv2.rectangle(frame, (width_start, height_start), (width_end, height_end), (255, 0, 0), 2)
                cv2.imshow("Yellow Mask", yellow_mask)
                cv2.imshow("Edges", edges)
                cv2.imshow("Original Frame", frame)
                cv2.waitKey(1)
            
            return width_start, width_end, height_start, height_end
    
    return None  # Return None if no sticker is detected

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # at each frame, crop the image to stabilize 
    crop_coords = find_base_region(frame)
    if crop_coords:
        w_start, w_end, h_start, h_end = crop_coords
        if print_messages:
            print(f"Width ({w_start}->{w_end})\nHeight ({h_start}->{h_end})")
        cropped_frame = frame[h_start:h_end, w_start:w_end]
        writer.write(cropped_frame)
        #Convert to HSV for red detection(red pen cap)
        hsv = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)

        # Define red color range in HSV(using high and low ranges for hue since red is split in half by the 0-180deg notation)
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        # Create masks for each red and then add them
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2

        # Find contours(pretty much same as last one)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get largest contour 
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)

            if M["m00"] != 0:
                # Compute center of the contour
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                trajectory.append((cx, cy))

                # Draw detected object
                cv2.circle(cropped_frame, (cx, cy), 5, (0, 255, 0), -1)
    
        #Draw the tracked path
        for i in range(1, len(trajectory)):
            cv2.line(cropped_frame, trajectory[i - 1], trajectory[i], (255, 255, 0), 3)

        cv2.imshow("Tracking", cropped_frame)
        writer.write(cropped_frame)

    #Adding keyboardinterrupt 
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
if writer:
    writer.release()
cv2.destroyAllWindows()

