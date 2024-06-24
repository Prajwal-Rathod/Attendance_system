from flask import Flask, render_template, request, redirect, url_for, make_response
import cv2
import numpy as np
import face_recognition
import os
import pandas as pd
from datetime import datetime, timedelta

app = Flask(__name__, template_folder=r"C:\Users\karth\OneDrive\Desktop\example\template")

# Define paths and variables for face recognition
known_faces_path = r"C:\Users\prajw\Desktop\example\encodings\finalattendence (2)\finalattendence\data_setofatt"
recognized_faces_path = r'C:\Users\prajw\Desktop\example\encodings\recognized_faces'
unknown_faces_path = r"C:\Users\prajw\Desktop\example\encodings\unknown_faces"

if not os.path.exists(recognized_faces_path):
    os.makedirs(recognized_faces_path)
if not os.path.exists(unknown_faces_path):
    os.makedirs(unknown_faces_path)

images = []
classNames = []
myList = os.listdir(known_faces_path)
print(myList)
for cl in myList:
    curImg = cv2.imread(os.path.join(known_faces_path, cl))
    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    else:
        print(f"Warning: Unable to load image {cl}")
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Open a video file instead of capturing from a webcam
video_file = r'C:\Users\prajw\Desktop\example\test4.mp4'  # Update with the path to your video file
cap = cv2.VideoCapture(video_file)

recognized_faces = set()  # Set to keep track of recognized faces

# Store the last time an unknown face was saved
last_unknown_save_time = datetime.min

# Sample user data (you should replace this with a database)
users = [{'username': 'user1', 'password': '1234'}]

# Define a variable to store the attendance data for this session
session_attendance = []

def markAttendance(name):
    now = datetime.now()
    dtString = now.strftime('%d:%m:%Y  %H:%M:%S')
    
    if not os.path.isfile('Attendance.csv'):
        with open('Attendance.csv', 'w') as f:
            f.write('Name,Time\n')
    
    df = pd.read_csv(r"C:\Users\karth\OneDrive\Desktop\example\encodings\Attendance.csv")
    if name not in df['Name'].values:
        new_row = pd.DataFrame({'Name': [name], 'Time': [dtString]})  # Fixed the column name 'Time'
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv('Attendance.csv', index=False)

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/dashboard', methods=['POST', 'GET'])
def dashboard():
    if request.method == 'POST':
        subject_name = request.form['username']
        password = request.form['password']

        for user in users:
            if user['username'] == subject_name and user['password'] == password:
                return render_template('dashboard.html')  # Redirect to the dashboard page

        return 'Login failed. Invalid subject name or password.'
    else:
        return render_template('dashboard.html')

@app.route('/start')
def take_attendance():
    global recognized_faces, last_unknown_save_time, session_attendance
    
    # Generate a unique CSV file name with a timestamp for each run
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    csv_file_name = f'Attendance_{timestamp}.csv'
    
    while True:
        success, img = cap.read()
        if not success:
            break
        
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace, tolerance=0.5)  # Adjust the tolerance value
            matchIndex = np.argmin(face_recognition.face_distance(encodeListKnown, encodeFace))

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                if name not in recognized_faces:
                    recognized_faces.add(name)
                    
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    markAttendance(name)
                    
                    recognized_filename = os.path.join(recognized_faces_path, f'{name}_{timestamp}.jpg')
                    cv2.imwrite(recognized_filename, img[y1:y2, x1:x2])
                    
                    # Append the attendance data to the session_attendance list
                    session_attendance.append((name, timestamp))
            else:
                # Check if enough time has passed since the last unknown face save
                if (datetime.now() - last_unknown_save_time) > timedelta(seconds=30):  # Change 30 to the desired threshold in seconds
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
                    cv2.putText(img, 'Unknown', (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    
                    unknown_filename = os.path.join(unknown_faces_path, f'unknown_{timestamp}.jpg')
                    cv2.imwrite(unknown_filename, img[y1:y2, x1:x2])
                    
                    last_unknown_save_time = datetime.now()  # Update the last unknown face save time

        cv2.imshow('Video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return redirect(url_for('dashboard'))

@app.route('/open-csv', methods=['GET'])
def open_csv():
    # Generate a unique CSV file name with a timestamp for each request
    timestamp = datetime.now().strftime("%d %m %Y %H %M %S")
    csv_file_name = f'Attendance_{timestamp}.csv'

    # Create and write data to the new CSV file
    with open(csv_file_name, 'w') as file:
        file.write('Name,Time\n')
        for name, time in session_attendance:
            file.write(f"{name},{time}\n")

    # Read the content of the new CSV file
    with open(csv_file_name, 'r') as file:
        csv_content = file.read()

    # Create a response with the CSV content and appropriate headers
    response = make_response(csv_content)
    response.headers['Content-Type'] = 'text/csv'
    response.headers['Content-Disposition'] = f'inline; filename={csv_file_name}'

    return response

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
