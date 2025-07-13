import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load model
model = load_model("models/emotion_model.h5")
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Start camera
cap = cv2.VideoCapture(0)

print("Press SPACE to capture emotion, ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    display_frame = frame.copy()

    # Draw instruction
    cv2.putText(display_frame, "Press SPACE to capture emotion", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Emotion Detection', display_frame)

    key = cv2.waitKey(1)

    if key == 27:  # ESC to exit
        break

    elif key == 32:  # SPACE to capture
        face = cv2.resize(gray, (48, 48))
        face = face.astype("float") / 255.0
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)

        preds = model.predict(face)
        label = emotion_labels[np.argmax(preds)]

        # Show prediction
        print(f"Detected Emotion: {label}")
        cv2.putText(frame, f"Emotion: {label}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        cv2.imshow("Emotion Detection", frame)
        cv2.waitKey(1500)  # Show result for 1.5 seconds

cap.release()
cv2.destroyAllWindows()
