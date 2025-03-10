import os
import cv2
import csv
import numpy as np
import face_recognition
from datetime import datetime

class AutomaticAttendance:
    def __init__(self, students_folder, attendance_file):
        """
        Initialize the attendance system
        
        Args:
            students_folder (str): Path to folder containing student images
            attendance_file (str): Path to CSV file where attendance will be recorded
        """
        self.students_folder = students_folder
        self.attendance_file = attendance_file
        self.known_face_encodings = []
        self.known_face_names = []
        self.already_marked = set()  # To prevent duplicate entries
        
        # Create attendance file if it doesn't exist
        if not os.path.exists(attendance_file):
            with open(attendance_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Name', 'Date', 'Time'])
    
    def load_student_images(self):
        """Load all student images from the folder and encode their faces"""
        print("Loading student images...")
        for filename in os.listdir(self.students_folder):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                # Extract student name from filename (assuming format like "John_Smith.jpg")
                student_name = os.path.splitext(filename)[0].replace('_', ' ')
                
                # Load image and encode face
                image_path = os.path.join(self.students_folder, filename)
                student_image = face_recognition.load_image_file(image_path)
                
                # Try to find a face in the image
                face_encodings = face_recognition.face_encodings(student_image)
                
                if len(face_encodings) > 0:
                    # Use the first face found in the image
                    face_encoding = face_encodings[0]
                    self.known_face_encodings.append(face_encoding)
                    self.known_face_names.append(student_name)
                    print(f"Loaded face for: {student_name}")
                else:
                    print(f"No face found in {filename}. Skipping...")
        
        print(f"Loaded {len(self.known_face_names)} student faces")
    
    def mark_attendance(self, name):
        """Record attendance for a student in the CSV file"""
        # Check if this student has already been marked today
        if name in self.already_marked:
            return
        
        now = datetime.now()
        date_string = now.strftime("%Y-%m-%d")
        time_string = now.strftime("%H:%M:%S")
        
        with open(self.attendance_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, date_string, time_string])
        
        self.already_marked.add(name)
        print(f"Marked attendance for {name}")
    
    def start_recognition(self, camera_source=0, detection_interval=30):
        """
        Start the facial recognition process using the webcam
        
        Args:
            camera_source (int): Camera index to use (default is 0 for the primary webcam)
            detection_interval (int): Number of frames between face detection attempts
        """
        if not self.known_face_encodings:
            print("No student faces loaded. Please load student images first.")
            return
        
        # Reset the already marked set at the start of a session
        self.already_marked = set()
        
        print("Starting facial recognition. Press 'q' to quit.")
        
        # Initialize webcam
        video_capture = cv2.VideoCapture(camera_source)
        
        if not video_capture.isOpened():
            print("Error: Could not open video source.")
            return
        
        frame_count = 0
        
        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()
            
            if not ret:
                print("Failed to grab frame from camera.")
                break
            
            # Only process every detection_interval frames to save CPU
            if frame_count % detection_interval == 0:
                # Resize frame for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                
                # Convert from BGR color (OpenCV) to RGB (face_recognition)
                rgb_small_frame = small_frame[:, :, ::-1]
                
                # Find all faces in the current frame
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                face_names = []
                for face_encoding in face_encodings:
                    # Compare with known faces
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
                    name = "Unknown"
                    
                    # Use the distance to find the best match
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]
                            
                            # Mark attendance for recognized student
                            self.mark_attendance(name)
                    
                    face_names.append(name)
                
                # Display the results
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    # Scale back up face locations since we scaled the image down
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    
                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    
                    # Draw a label with the name below the face
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
            
            # Display the resulting frame
            cv2.imshow('Automatic Attendance System', frame)
            
            # Exit loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()

def main():
    # Directory structure
    students_folder = "student_images"
    attendance_file = "attendance.csv"
    
    # Create directories if they don't exist
    if not os.path.exists(students_folder):
        os.makedirs(students_folder)
        print(f"Created {students_folder} directory. Please add student images.")
    
    # Initialize the attendance system
    attendance_system = AutomaticAttendance(students_folder, attendance_file)
    
    # Load known student faces
    attendance_system.load_student_images()
    
    # Start facial recognition
    attendance_system.start_recognition()

if __name__ == "__main__":
    main()