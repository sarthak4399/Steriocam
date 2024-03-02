import cv2

# Initialize camera streams
# For the left camera feed from DroidCam
left_droidcam_ip_address = '192.168.1.2'
left_droidcam_port = '4747'
left_droidcam_url = f'http://{left_droidcam_ip_address}:{left_droidcam_port}/video'
cap_left_droidcam = cv2.VideoCapture(left_droidcam_url)

# For the right camera feed from DroidCam
right_droidcam_ip_address = '192.168.1.7'
right_droidcam_port = '4747'
right_droidcam_url = f'http://{right_droidcam_ip_address}:{right_droidcam_port}/video'
cap_right_droidcam = cv2.VideoCapture(right_droidcam_url)

# Set up stereo matcher
stereo = cv2.StereoSGBM_create(
    minDisparity=0, 
    numDisparities=16, 
    blockSize=9,
    preFilterCap=63, # Adjust if necessary
    uniquenessRatio=10, # Adjust if necessary
    speckleWindowSize=100, # Adjust if necessary
    speckleRange=32 # Adjust if necessary
)

while True:
    # Capture frames from both cameras
    ret1, frame_left = cap_left_droidcam.read()
    ret2, frame_right = cap_right_droidcam.read()

    if not (ret1 and ret2):
        break

    # Resize frames if needed (to match resolutions)
    # Rectify the images if needed
    # Compute disparity map
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
    disparity = stereo.compute(gray_left, gray_right)

    # Normalize the disparity map for better visualization
    disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Display the frames and disparity map
    cv2.imshow('Left DroidCam Feed', frame_left)
    cv2.imshow('Right DroidCam Feed', frame_right)
    cv2.imshow('Disparity Map', disparity_normalized)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap_left_droidcam.release()
cap_right_droidcam.release()
cv2.destroyAllWindows()
