import os
import cv2
import numpy as np
import face_recognition
from keras.models import load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Model
from keras.models import Sequential

# Load emotion recognition model
emotion_model = load_model("/Users/alekhya/Desktop/Robotics and AI/Semester - 5/VSI/CW2/emo_model.h5")
emotions = ('Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise')

#Training the model
emotion_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Debugging the model
emotion_model.summary()

# Loading Face Recognition Data
data_location = '/Users/alekhya/Desktop/Robotics and AI/Semester - 5/VSI/CW2/ref pics'

# Load data location and known entities
def load_data_entities(data_location):
    entity_encodings = []
    entity_labels = []

    for file_name in os.listdir(data_location):
        if not file_name.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            continue

        image = face_recognition.load_image_file(os.path.join(data_location, file_name))
        face_encodings = face_recognition.face_encodings(image)
        
        if len(face_encodings) > 0:
            entity_encodings.append(face_encodings[0])
            label = os.path.splitext(file_name)[0]
            entity_labels.append(label)

    return entity_encodings, entity_labels

entity_encodings, entity_labels = load_data_entities(data_location)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Adjust scale factor for face detection
scale_factor = 1.0

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: No frame captured or frame is empty.")
        continue

    # Face recognition part
    small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    processed_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    detected_face_locations = face_recognition.face_locations(processed_frame)
    detected_face_encodings = face_recognition.face_encodings(processed_frame, detected_face_locations)

    detected_entity_labels = []
    for face_encoding in detected_face_encodings:
        matches = face_recognition.compare_faces(entity_encodings, face_encoding)
        label = "Unknown"

        face_distances = face_recognition.face_distance(entity_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            label = entity_labels[best_match_index]

        detected_entity_labels.append(label)

    # Emotion Analysis part
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for (top, right, bottom, left), label in zip(detected_face_locations, detected_entity_labels):
        # Scale back up face locations since the frame detected was scaled
        top = int(top / scale_factor)
        right = int(right / scale_factor)
        bottom = int(bottom / scale_factor)
        left = int(left / scale_factor)

        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), thickness=2)
        ROI = grayscale_frame[top:bottom, left:right]
        
        # Check if ROI is not empty before resizing
        if ROI.size != 0:
            ROI = cv2.resize(ROI, (48, 48))
            ROI = np.expand_dims(ROI, axis=-1)
            ROI = np.expand_dims(ROI, axis=0)
            ROI = ROI / 255.0

            predictions = emotion_model.predict(ROI)
            max_index = np.argmax(predictions[0])
            predicted_emotion = emotions[max_index]

            cv2.putText(frame, f"{label}: {predicted_emotion}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Facial Analysis', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
