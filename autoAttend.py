import os
import cv2
import csv
import numpy as np
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
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.student_ids = []  # Maps index to student name
        self.already_marked = set()  # To prevent duplicate entries
        self.model_trained = False
        
        # Create attendance file if it doesn't exist
        if not os.path.exists(attendance_file):
            with open(attendance_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Name', 'Date', 'Time'])
    
    def load_student_images(self):
        """Load all student images from the folder and train the face recognizer"""
        print("Loading student images and training the recognizer...")
        
        faces = []
        labels = []
        label_counter = 0
        
        for person_name in os.listdir(self.students_folder):
            person_path = os.path.join(self.students_folder, person_name)
            
            if os.path.isdir(person_path):
                self.student_ids.append(person_name.replace('_', ' '))
                
                for image_name in os.listdir(person_path):
                    if image_name.endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(person_path, image_name)
                        
                        # Read the image in grayscale
                        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        
                        if img is None:
                            print(f"Error loading image {image_path}")
                            continue
                        
                        # Detect faces in the image
                        faces_rect = self.face_detector.detectMultiScale(
                            img, scaleFactor=1.1, minNeighbors=4
                        )
                        
                        if len(faces_rect) > 0:
                            (x, y, w, h) = faces_rect[0]  # Use the first face found
                            face_img = img[y:y+h, x:x+w]
                            
                            # Add face and label to training data
                            faces.append(face_img)
                            labels.append(label_counter)
                            print(f"Loaded face from {image_path}")
                        else:
                            print(f"No face found in {image_path}. Skipping...")
                
                label_counter += 1
        
        if len(faces) == 0:
            print("No faces were detected in any images. Please check the images and folder structure.")
            return False
        
        # Train the face recognizer
        print(f"Training the recognizer with {len(faces)} images...")
        self.face_recognizer.train(faces, np.array(labels))
        self.model_trained = True
        print(f"Face recognizer trained with {label_counter} students")
        return True
    
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
    
    def start_recognition(self, camera_source=0, detection_interval=5):
        """
        Start the facial recognition process using the webcam
        
        Args:
            camera_source (int): Camera index to use (default is 0 for the primary webcam)
            detection_interval (int): Number of frames between face detection attempts
        """
        if not self.model_trained:
            print("Face recognizer not trained. Please load student images first.")
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
            
            # Create a copy for display
            display_frame = frame.copy()
            
            # Only process every detection_interval frames to save CPU
            if frame_count % detection_interval == 0:
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                
                for (x, y, w, h) in faces:
                    # Extract the face ROI
                    face_roi = gray[y:y+h, x:x+w]
                    
                    # Perform recognition
                    try:
                        label, confidence = self.face_recognizer.predict(face_roi)
                        
                        # Higher confidence means less certainty in LBPH
                        confidence_threshold = 70
                        
                        if confidence < confidence_threshold and label < len(self.student_ids):
                            name = self.student_ids[label]
                            
                            # Mark attendance
                            self.mark_attendance(name)
                            
                            # Display name and confidence
                            confidence_text = f"{round(100 - confidence)}%"
                        else:
                            name = "Unknown"
                            confidence_text = ""
                        
                        # Draw a box around the face
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                        
                        # Put text below the face
                        cv2.putText(display_frame, name, (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        cv2.putText(display_frame, confidence_text, (x, y+h+50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    except Exception as e:
                        print(f"Error during recognition: {e}")
            
            # Display the resulting frame
            cv2.imshow('Automatic Attendance System', display_frame)
            
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
        print(f"Created {students_folder} directory. Please add student image folders.")
        print("Each student should have their own folder named after them.")
    
    # Initialize the attendance system
    attendance_system = AutomaticAttendance(students_folder, attendance_file)
    
    # Load known student faces and train recognizer
    if attendance_system.load_student_images():
        # Start facial recognition
        attendance_system.start_recognition()
    else:
        print("Failed to train the face recognizer. Please check your images.")

if __name__ == "__main__":
    main()