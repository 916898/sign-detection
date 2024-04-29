import cv2
import numpy as np
from keras.models import load_model
import time

avg_predictions = []

# Load the trained model
model = load_model('smnist.h5')

# Function to preprocess the input image or frame from video
def preprocess_input(input_data):
    # Convert to grayscale
    gray = cv2.cvtColor(input_data, cv2.COLOR_BGR2GRAY)
    # Resize the image to 28x28 (size used in training)
    resized = cv2.resize(gray, (28, 28))
    # Reshape to match the input shape of the model
    processed_data = resized.reshape(1, 28, 28, 1)
    # Normalize the data
    processed_data = processed_data / 255.0
    return processed_data

# Function to detect hand sign
def detect_hand_sign(input_data):
    # Preprocess the input data
    processed_data = preprocess_input(input_data)
    # Predict the class of the hand sign
    prediction = model.predict(processed_data)
    # Get the index of the class with the highest probability
    predicted_class = np.argmax(prediction)
    # Return the predicted class label
    return predicted_class

# Function to draw predicted class label on the image
def draw_prediction(image, predicted_class):
    # Define class labels (assuming your classes are labeled as 0 to 23)
    class_labels = ["J", "B", "C", "S", "E", "F", "G", "H", "I", "A", "K", "L", "M", 
                    "N", "O", "P", "Q", "R", "D", "T", "U", "V", "W", "X", "Y", "Z"]

    predicted_label = class_labels[predicted_class]
    # Draw the predicted class label on the image
    cv2.putText(image, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return image

# Function to process input (image or video frame) and display output
def process_input(input_data):
    global avg_predictions
    if isinstance(input_data, str):  # Check if input is a filename (assumes image or video)
        if input_data.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # Image file
            image = cv2.imread(input_data)
            predicted_class = detect_hand_sign(image)
            output_image = draw_prediction(image, predicted_class)
            cv2.imshow('Hand Sign Detection', output_image)
            cv2.waitKey(10000)  # Display image for 10 seconds (10000 milliseconds)
            cv2.destroyAllWindows()  # Close the window after 10 seconds
        elif input_data.lower().endswith(('.mp4', '.gif', '.m4a')):  # Video file
            video_capture = cv2.VideoCapture(input_data)
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    break
                predicted_class = detect_hand_sign(frame)
                avg_predictions.append(predicted_class)
                output_frame = draw_prediction(frame, predicted_class)
                cv2.imshow('Hand Sign Detection', output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            video_capture.release()
        else:
            print("Unsupported file format.")
    else:  # Input is an image or video frame
        predicted_class = detect_hand_sign(input_data)
        avg_predictions.append(predicted_class)
        output_data = draw_prediction(input_data, predicted_class)
        cv2.imshow('Hand Sign Detection', output_data)
        cv2.waitKey(0)

# Example usage for image input
input_image = 'test/e.jpg'  # Replace with the path to your image
process_input(input_image)

# Example usage for video input
# input_video = 'your_video.mp4'  # Replace with the path to your video

# process_input(input_video)

# Calculate the most frequent prediction (mode) over all frames
if avg_predictions:
    mode_prediction = max(set(avg_predictions), key=avg_predictions.count)
    print("Most frequent prediction:", mode_prediction)
