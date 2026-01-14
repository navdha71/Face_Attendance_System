import cv2                     # Import OpenCV for webcam and image processing
import numpy as np                 # Import NumPy for numerical calculations
import pandas as pd             # Import Pandas to create and save attendance CSV
import datetime                # Import datetime to get current time
from tensorflow.keras.models import load_model  # Import function to load trained CNN model

# Load model and labels
model = load_model("model/face_model.h5")        # Load the trained face recognition model
label_dict = np.load("model/labels.npy", allow_pickle=True).item()
                                              # Load label dictionary (number → person name)

# Open webcam
cap = cv2.VideoCapture(0)                  # Start the laptop webcam
attendance = []                            # List to store names for attendance

while True:                            # Start infinite loop for live camera feed
    ret, frame = cap.read()                 # Capture a frame from the webcam
    if not ret:                          # If frame is not captured properly
        break                         # Exit the loop

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    # Convert color image to grayscale
    face = cv2.resize(gray, (100, 100))         
                                     # Resize image to match training image size
    face = face.reshape(1, 100, 100, 1) / 255.0 
                                             # Reshape image for CNN input
                                             # Normalize pixel values (0–255 → 0–1)

    prediction = model.predict(face, verbose=0) # Predict the face using trained CNN model

    label = np.argmax(prediction)            # Get index of highest prediction value
    name = label_dict[label]                 # Convert label number to person name

    # Display name on webcam
    cv2.putText(frame, name, (50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                                            # Write the recognized name on the video frame

    cv2.imshow("Face Attendance", frame)  # Show the webcam window with name displayed


    # Mark attendance
    if name not in attendance:                     # Check if name is not already marked
        attendance.append(name)                       # Add name to attendance list

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):          # Check if user presses 'q'
        break                                    # Exit the loop

cap.release()                         # Turn off the webcam
cv2.destroyAllWindows()              # Close all OpenCV windows

# Save attendance in CSV
df = pd.DataFrame(attendance, columns=["Name"])        # Create a DataFrame with names
df["Time"] = datetime.datetime.now().strftime("%H:%M:%S")  
                                                   # Add current time column to attendance
df.to_csv("attendance/attendance.csv", index=False)  # Save attendance data to CSV file

print("Attendance marked successfully")      # Print success message
