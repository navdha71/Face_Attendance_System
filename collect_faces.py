import cv2              # Import OpenCV library to work with camera, images, and video
import os               # Import OS library to work with folders and file paths

name = input("Enter person name: ")      # Ask the user to enter the person's name
dataset_path = "dataset/" + name         # Create a folder path using the person's name

if not os.path.exists(dataset_path):     # Check if the folder does NOT already exist
    os.makedirs(dataset_path)            # Create the folder to store images

cap = cv2.VideoCapture(0)             # Turn on the default webcam (0 means laptop camera)
count = 0                             # Initialize image counter to 0

while True:                           # Start an infinite loop
    ret, frame = cap.read()          
                                      # Capture one frame from the webcam
                                      # ret = True if frame is captured successfully
                                      # frame = actual image from the camera
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
                                           # Convert the captured color image into grayscale
    cv2.imshow("Face Capture", frame)  
                                     # Show the camera feed in a window named "Face Capture"
    if cv2.waitKey(1) & 0xFF == ord('c'):        
                                      # Check if the user presses the 'c' key
                                      # waitKey(1) waits for 1 millisecond for key press
                                      # ord('c') means ASCII value of 'c'
        cv2.imwrite(f"{dataset_path}/{count}.jpg", gray) 
                                              # Save the grayscale image in the folder
                                              # File name will be 0.jpg, 1.jpg, 2.jpg, etc.
        count += 1                      # Increase image count by 1
        print("Image saved:", count)    # Print how many images have been saved

    if count == 50:             # Check if 50 images are saved
        break                   # Stop the loop once 50 images are captured

cap.release()                   # Turn off the webcam
cv2.destroyAllWindows()         # Close all OpenCV windows

