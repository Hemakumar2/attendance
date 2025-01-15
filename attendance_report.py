import mysql.connector

# Connect to MySQL database
db = mysql.connector.connect(
    host="your-hostname",
    user="your-username",
    password="your-password",
    database="attendance_system"
)
cursor = db.cursor()

# Get attendance count and calculate percentage
cursor.execute("""
    SELECT students.id, students.name, COUNT(attendance.id) AS total_attendance
    FROM students
    LEFT JOIN attendance ON students.id = attendance.student_id
    GROUP BY students.id, students.name
""")
records = cursor.fetchall()

# Total sessions for percentage calculation (adjust this as needed)
total_sessions = 10

print("ID | Name       | Attendance (%)")
print("--------------------------------")
for record in records:
    student_id, name, total_attendance = record
    percentage = (total_attendance / total_sessions) * 100
    print(f"{student_id} | {name:10} | {percentage:.2f}%")

cursor.close()
db.close()
