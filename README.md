# 📁 Emotion-Detection-System

This project is a real-time facial emotion recognition system built with TensorFlow/Keras. This is **Version 1** of the system that uses a custom Convolutional Neural Network (CNN) to classify facial expressions into one of seven basic emotions.

The system can:

Train a model from scratch using labeled images.

Evaluate the model’s accuracy with test data and a confusion matrix.

Perform real-time emotion detection using your computer’s webcam.

🧠 Emotion Classes 

The model recognizes the following 7 emotions:

- Angry

- Disgust

- Fear

- Happy

- Sad

- Surprise

- Neutral


📂 Project Structure

emotion-detection/
│
├── data/                  # Contains training and testing image folders
│   ├── train/             # Subfolders per emotion (angry, happy, etc.)
│   └── test/              # Subfolders per emotion
│
├── models/                # Saved trained model files (.h5)
│
├── train.py               # Script to train the CNN model
├── validate.py            # Script to evaluate model accuracy + confusion matrix
├── main.py                # Real-time webcam emotion detection script
├── requirements.txt       # Project dependencies


⚙️ Setup Instructions

1. Clone the repository and navigate into the project folder
   
2. Create and activate a virtual environment
   
3. Install dependencies
   
   pip install -r requirements.txt


🚀 How to Run

Train the model:

python train.py

This saves the trained model as models/emotion_model.h5.

Validate the model:

python validate.py

Prints test accuracy, confusion matrix, and classification report.

Real-time emotion detection:

python main.py

Launches webcam and predicts emotion from live video feed.



📈 Model Architecture (CNN)

Input shape: 48x48 grayscale images

Layers:

  Conv2D → MaxPooling → Conv2D → MaxPooling
  
  Flatten → Dense → Dropout
  
  Dense (Softmax)
  
Loss: categorical_crossentropy

Optimizer: adam

Metrics: accuracy



✅ Sample Output

During training:
  
   - Accuracy and loss printed per epoch.
    
During validation:

   - Classification accuracy
   
   - Confusion matrix
   
During real-time:

   - Live camera feed with face detection
   
   - Emotion label shown above detected face



📄 License

This project is for learning purposes and may be adapted or extended freely.

