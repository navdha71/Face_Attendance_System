import cv2
import numpy as np
import pandas as pd
import datetime
from tensorflow.keras.models import load_model

# Load model and labels
model = load_model("model/face_model.h5")
label_dict = np.load("model/labels.npy", allow_pickle=True).item()

# Open webcam
cap = cv2.VideoCapture(0)
attendance = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (100, 100))
    face = face.reshape(1, 100, 100, 1) / 255.0

    prediction = model.predict(face, verbose=0)

    label = np.argmax(prediction)
    name = label_dict[label]

    # Display name on webcam
    cv2.putText(frame, name, (50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Face Attendance", frame)


    # Mark attendance
    if name not in attendance:
        attendance.append(name)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save attendance in CSV
df = pd.DataFrame(attendance, columns=["Name"])
df["Time"] = datetime.datetime.now().strftime("%H:%M:%S")
df.to_csv("attendance/attendance.csv", index=False)

print("Attendance marked successfully")
