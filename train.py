# train.py
import os
import numpy as np
# import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# Paths
train_dir = 'data/train'
model_save_path = 'models/emotion_model.h5'

# Data augmentation
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=30,
                                   zoom_range=0.2, horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(48, 48),
                                                    color_mode='grayscale', batch_size=64,
                                                    class_mode='categorical')

# Model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotion classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Save best model
checkpoint = ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_accuracy', mode='max')

# Train
model.fit(train_generator, epochs=20, callbacks=[checkpoint])
