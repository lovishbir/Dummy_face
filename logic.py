#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import mediapipe as mp
import cv2

# Load your dataset (replace with actual data loading code)
# X_train, y_train_skin_disease, y_train_skin_type = load_data()
# X_test, y_test_skin_disease, y_test_skin_type = load_data()

# Example: You can use placeholder data as follows for testing purposes
# Replace with actual data loading and preprocessing
X_train = np.random.rand(100, 224, 224, 3)  # 100 sample images (224x224x3)
y_train_skin_disease = np.random.randint(0, 5, 100)  # 5 categories for skin diseases (dummy data)
y_train_skin_type = np.random.randint(0, 4, 100)  # 4 categories for skin types (dummy data)
X_test = np.random.rand(20, 224, 224, 3)  # 20 test sample images (224x224x3)
y_test_skin_disease = np.random.randint(0, 5, 20)
y_test_skin_type = np.random.randint(0, 4, 20)

# One-hot encode the labels
y_train_skin_disease = tf.keras.utils.to_categorical(y_train_skin_disease, num_classes=5)
y_train_skin_type = tf.keras.utils.to_categorical(y_train_skin_type, num_classes=4)
y_test_skin_disease = tf.keras.utils.to_categorical(y_test_skin_disease, num_classes=5)
y_test_skin_type = tf.keras.utils.to_categorical(y_test_skin_type, num_classes=4)

# Define the model
input_layer = layers.Input(shape=(224, 224, 3))  # Example input shape (224x224x3 for RGB images)

# Convolutional layers for feature extraction
x = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(128, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)

# Flatten the output from the convolutional layers
x = layers.Flatten()(x)

# Dense layer for classification
x = layers.Dense(512, activation='relu')(x)

# Define the output layers
num_diseases = 5  # Adjust based on your dataset
num_skin_types = 4  # Adjust based on your dataset

skin_disease_output = layers.Dense(num_diseases, activation='softmax', name='skin_disease')(x)
skin_type_output = layers.Dense(num_skin_types, activation='softmax', name='skin_type')(x)

# Create the model with two outputs
model = Model(inputs=input_layer, outputs=[skin_disease_output, skin_type_output])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss={'skin_disease': 'categorical_crossentropy', 'skin_type': 'categorical_crossentropy'}, 
              metrics={'skin_disease': ['accuracy'], 'skin_type': ['accuracy']})

# Train the model
model.fit(X_train, 
          {'skin_disease': y_train_skin_disease, 'skin_type': y_train_skin_type},
          epochs=10, batch_size=32, 
          validation_data=(X_test, {'skin_disease': y_test_skin_disease, 'skin_type': y_test_skin_type}))

# Use webcam for live detection (optional step)
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the BGR image to RGB for Mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw landmarks on the face
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
                
                # Get face bounding box from landmarks (example: around the eyes region)
                h, w, _ = frame.shape
                x_min = int(min([lm.x for lm in face_landmarks.landmark]) * w)
                y_min = int(min([lm.y for lm in face_landmarks.landmark]) * h)
                x_max = int(max([lm.x for lm in face_landmarks.landmark]) * w)
                y_max = int(max([lm.y for lm in face_landmarks.landmark]) * h)
                
                # Crop the face region (using the bounding box)
                face_roi = frame[y_min:y_max, x_min:x_max]
                
                # Resize face region to the input shape of the model (224x224)
                face_roi_resized = cv2.resize(face_roi, (224, 224))
                
                # Normalize the face image
                face_roi_resized = face_roi_resized / 255.0
                
                # Add a batch dimension
                face_roi_resized = np.expand_dims(face_roi_resized, axis=0)
                
                # Predict the skin disease and skin type
                skin_disease_pred, skin_type_pred = model.predict(face_roi_resized)
                
                # Get the predicted classes (index of max probability)
                skin_disease_class = np.argmax(skin_disease_pred)
                skin_type_class = np.argmax(skin_type_pred)
                
                # Map the prediction to label (if needed)
                skin_disease_labels = ['Acne', 'Eczema', 'Psoriasis', 'Melanoma', 'Rosacea']  # Update with actual labels
                skin_type_labels = ['Oily', 'Dry', 'Combination', 'Normal']  # Update with actual labels
                
                # Display the predictions on the frame
                cv2.putText(frame, f'Skin Disease: {skin_disease_labels[skin_disease_class]}', 
                            (x_min, y_min - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame, f'Skin Type: {skin_type_labels[skin_type_class]}', 
                            (x_min, y_min - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display the result
        cv2.imshow('Skin Disease and Skin Type Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


# In[ ]:




