import cv2
import os

name = input("Enter person name: ")
dataset_path = "dataset/" + name

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Face Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('c'):
        cv2.imwrite(f"{dataset_path}/{count}.jpg", gray)
        count += 1
        print("Image saved:", count)

    if count == 50:
        break

cap.release()
cv2.destroyAllWindows()

