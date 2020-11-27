import face_recognition   #DLIB
import cv2    #OPENCV
#import numpy as np 

cam  = cv2.VideoCapture(0)   #open camira
image_paths = ["img1.jpg", "img2.jpg" ] 
known_face_names = ["Norhan", 'radwa']
known_face_encodings = []
for image_path in image_paths:
    face = face_recognition.load_image_file(image_path)
    face_encoding = face_recognition.face_encodings(face)[0]
    known_face_encodings.append(face_encoding)

while True:
    ret, frame = cam.read()
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    for i in range(len(face_locations)):
        top, right, bottom, left = face_locations[i]
        face_encoding = face_encodings[i]

        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        print(matches) # array of bool var
        name = "Unknown"
        # check the best match 
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        min_value = 99999999999999999
        min_index = -1
        for i in range(len(face_distances)):
            if face_distances[i]< min_value:
                min_value = face_distances[i]
                min_index = i
        if matches[min_index] == True:
            name = known_face_names[min_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2) 
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
    cv2.imshow('viewer', frame)
    key = cv2.waitKey(100)
    if  key == ord('q'):
        break

# Release handle to the webcam
cam.release()
cv2.destroyAllWindows()
