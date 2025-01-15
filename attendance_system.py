import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mysql.connector
from datetime import date

# Load the trained model and labels
model = load_model("keras_Model.h5", compile=False)
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]

# Connect to MySQL database
db = mysql.connector.connect(
    host="your-hostname",  # Replace with your MySQL hostname
    user="your-username",  # Replace with your MySQL username
    password="your-password",  # Replace with your MySQL password
    database="attendance_system"  # Replace with your database name
)
cursor = db.cursor()


# Function to log attendance
def log_attendance(face_label):
    cursor.execute("SELECT id FROM students WHERE face_label = %s", (face_label,))
    result = cursor.fetchone()
    if result:
        student_id = result[0]
        today = date.today()
        # Check if attendance is already logged today
        cursor.execute("SELECT * FROM attendance WHERE student_id = %s AND timestamp = %s", (student_id, today))
        if not cursor.fetchone():
            cursor.execute("INSERT INTO attendance (student_id) VALUES (%s)", (student_id,))
            db.commit()
            print(f"Attendance logged for {face_label}.")
        else:
            print(f"Attendance already recorded for {face_label}.")
    else:
        print(f"No record found for {face_label}.")


# Start webcam feed
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Error: Could not access the camera.")
    exit()

print("Press ESC to exit.")
while True:
    ret, frame = camera.read()
    if not ret:
        print("Failed to capture frame. Exiting.")
        break

    # Resize the frame and preprocess it
    resized_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    normalized_frame = np.asarray(resized_frame, dtype=np.float32).reshape(1, 224, 224, 3)
    normalized_frame = (normalized_frame / 127.5) - 1  # Normalize

    # Make predictions
    predictions = model.predict(normalized_frame)
    index = np.argmax(predictions)
    face_label = class_names[index]
    confidence_score = predictions[0][index]

    # Log attendance if confidence is high
    if confidence_score > 0.9:  # Adjust the threshold if necessary
        log_attendance(face_label)

    # Display the webcam feed with predictions
    cv2.putText(frame, f"{face_label} ({confidence_score:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow("Attendance System", frame)

    # Exit on pressing ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

camera.release()
cv2.destroyAllWindows()
cursor.close()
db.close()
