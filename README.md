Attendance System Using Teachable Machine, Python, and MySQL Online:
This project is a real-time attendance system that uses a Teachable Machine model for face recognition, stores attendance data in an online MySQL database, and calculates attendance percentages for each individual.

Table of Contents:
Overview
Features
Technologies Used
Step-by-Step Procedure
1. Teachable Machine
2. MySQL Online Setup
3. Python Code Explanation
How to Run
Database Schema
Project Structure
Acknowledgements

Overview:
This system leverages machine learning and a database to automate attendance recording and generate attendance percentages. It recognizes individuals using a Teachable Machine model and updates attendance records in a MySQL database.

Features:
Recognizes faces for four classes (e.g., students).
Tracks and stores attendance in a MySQL online database.
Calculates and displays attendance percentages.
Displays class names, confidence scores, and attendance records in real-time.

Technologies Used:
Teachable Machine: For training and exporting the face recognition model.
Python: For integrating all components.
TensorFlow/Keras: For loading and using the trained model.
OpenCV: For webcam-based image capture and processing.
MySQL Online: For storing and managing attendance data.

Step-by-Step Procedure:

Teachable Machine
![image](https://github.com/user-attachments/assets/f0aea921-ed0f-4ed8-a3bb-fc814b1e0310)

Visit Teachable Machine.
![image](https://github.com/user-attachments/assets/f1b39a71-7615-4ac3-a387-b3349aa37340)

Select Image Project and create a new project.
![image](https://github.com/user-attachments/assets/f4b4b180-ea46-49d1-866c-0767ddd5589d)


Add four classes:
Class 1: Add images of the first individual.
Class 2: Add images of the second individual.
Class 3: Add images of the third individual.
Class 4: Add images of the fourth individual.
![image](https://github.com/user-attachments/assets/2590d979-0179-48b9-9f99-f285a72d9ef8)

Train the model by clicking the Train Model button.
Export the trained model:
Select Keras Model as the export format.
![image](https://github.com/user-attachments/assets/a0b9e877-77eb-4e56-923a-97d1001bd717)

Download the keras_Model.h5 and labels.txt files.

MySQL Online Setup:
Choose an online MySQL service (e.g., PlanetScale, Railway).
Create a MySQL database named attendance_db.
Run the following SQL commands to set up the tables:

CREATE DATABASE attendance_db;
USE attendance_db;

CREATE TABLE students (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    face_label VARCHAR(255) NOT NULL
);

CREATE TABLE attendance (
    id INT AUTO_INCREMENT PRIMARY KEY,
    student_id INT NOT NULL,
    timestamp DATE DEFAULT CURRENT_DATE,
    FOREIGN KEY (student_id) REFERENCES students(id)
);

INSERT INTO students (name, face_label) 
VALUES 
    ('Hema', 'Class 1'),
    ('Divya', 'Class 2'),
    ('Kavya', 'Class 3'),
    ('Manisha', 'Class 4');

Here is a step-by-step explanation of the coding procedure for creating an attendance system using Python, Teachable Machine, and MySQL. This guide will break down the code into manageable sections to help you understand each component.

Step 1: Set Up Your Python Environment

Install Required Libraries: Install the libraries needed for the project:
pip install tensorflow opencv-python mysql-connector-python numpy

TensorFlow: To load and use the trained model from Teachable Machine.
OpenCV: For capturing and processing webcam images.
MySQL Connector: To connect and interact with the MySQL database.
NumPy: For numerical computations and image array processing.
![image](https://github.com/user-attachments/assets/945c4709-dc73-4763-b57d-6a611c1479d5)


Step 2: Connect to the MySQL Database
First, set up the connection to your MySQL database.
import mysql.connector

# Connect to MySQL database
db = mysql.connector.connect(
    host="your-host",        # Replace with your MySQL host
    user="your-username",    # Replace with your MySQL username
    password="your-password",# Replace with your MySQL password
    database="attendance_db" # Replace with your database name
)

cursor = db.cursor()
This code establishes a connection with the database and initializes a cursor to execute SQL commands.

Step 3: Load the Trained Model
Load the model and labels generated from Teachable Machine.
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]

keras_Model.h5: The trained model file.
labels.txt: The text file containing class labels corresponding to the model's output.

Step 4: Initialize the Webcam
Use OpenCV to capture video feed from the webcam.
import cv2

# Initialize webcam
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Could not access the webcam.")
    exit()
The cv2.VideoCapture(0) command initializes the webcam. The 0 indicates the default camera. Ensure your camera is connected.

Step 5: Process the Webcam Feed
Capture images from the webcam, preprocess them for the model, and make predictions.
import numpy as np

while True:
    ret, frame = camera.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Preprocess the image
    image = cv2.resize(frame, (224, 224))                # Resize to model input size
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1                          # Normalize the image

    # Make a prediction
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = round(prediction[0][index] * 100, 2)

    # Display prediction on the screen
    cv2.putText(frame, f"Class: {class_name}, Confidence: {confidence_score}%", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Webcam", frame)

    # Exit on pressing ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

This code processes each frame:
Resizes it to the required input size for the model.
Normalizes the image data.
Makes a prediction using the model.
Displays the predicted class and confidence score.

Step 6: Update Attendance in the Database:

# Update attendance in the database
cursor.execute("SELECT id FROM students WHERE face_label = %s", (class_name,))
result = cursor.fetchone()
if result:
    student_id = result[0]
    cursor.execute("INSERT INTO attendance (student_id) VALUES (%s)", (student_id,))
    db.commit()
Check if the predicted class (class_name) exists in the students table.
Insert a new row in the attendance table with the corresponding student_id.

Step 7: Cleanup Resources
camera.release()
cv2.destroyAllWindows()
cursor.close()
db.close()

Step 8: Full Code
Here’s the complete Python script:

from tensorflow.keras.models import load_model
import cv2
import numpy as np
import mysql.connector

# Connect to the database
db = mysql.connector.connect(
    host="your-host",
    user="your-username",
    password="your-password",
    database="attendance_db"
)
cursor = db.cursor()

# Load the model and labels
model = load_model("keras_Model.h5", compile=False)
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]

# Initialize the webcam
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Main loop
while True:
    ret, frame = camera.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Preprocess the image
    image = cv2.resize(frame, (224, 224))
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1

    # Make a prediction
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = round(prediction[0][index] * 100, 2)

    # Display prediction
    cv2.putText(frame, f"Class: {class_name}, Confidence: {confidence_score}%", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Webcam", frame)

    # Update attendance
    cursor.execute("SELECT id FROM students WHERE face_label = %s", (class_name,))
    result = cursor.fetchone()
    if result:
        student_id = result[0]
        cursor.execute("INSERT INTO attendance (student_id) VALUES (%s)", (student_id,))
        db.commit()

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
camera.release()
cv2.destroyAllWindows()
cursor.close()
db.close()

Create one more file attendance_report.py
import mysql.connector

# Connect to the database
db = mysql.connector.connect(
    host="your-host",        # Replace with your MySQL host
    user="your-username",    # Replace with your MySQL username
    password="your-password",# Replace with your MySQL password
    database="attendance_db" # Replace with your database name
)
cursor = db.cursor()

# Fetch all students and their attendance count
cursor.execute("""
    SELECT 
        students.id, students.name, COUNT(attendance.student_id) AS days_present
    FROM 
        students
    LEFT JOIN 
        attendance ON students.id = attendance.student_id
    GROUP BY 
        students.id
""")
attendance_data = cursor.fetchall()

# Calculate total days (unique dates in attendance table)
cursor.execute("SELECT COUNT(DISTINCT timestamp) FROM attendance")
total_days = cursor.fetchone()[0]

# Display attendance report
print("Attendance Report:")
print(f"{'ID':<5}{'Name':<15}{'Days Present':<15}{'Attendance %':<10}")
print("-" * 45)

for student_id, name, days_present in attendance_data:
    percentage = (days_present / total_days) * 100 if total_days > 0 else 0
    print(f"{student_id:<5}{name:<15}{days_present:<15}{percentage:.2f}%")

# Cleanup
cursor.close()
db.close()

Folder Structure:
|-- attendance.py
|-- attendance_report.py
|-- keras_Model.h5
|-- labels.txt
|-- README.md


Step 9: Run the Script:
Ensure keras_Model.h5 and labels.txt are in the same folder as the script.
Run the script:
python attendance.py

Customization
Add more students by inserting records into the students table.
Modify the attendance_report.py script to save reports to a file (e.g., CSV or Excel).
