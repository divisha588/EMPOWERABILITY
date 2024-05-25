from keras.models import load_model
import cv2
import numpy as np
import pyttsx3

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model(r"C:\Users\divisha\Downloads\money counter\converted_keras (1)\keras_model.h5", compile=False)

# Load the labels
class_names = open(r"C:\Users\divisha\Downloads\money counter\converted_keras (1)\labels.txt", "r").readlines()

# Initialize pyttsx3
engine = pyttsx3.init()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Speak out the identified currency
    currency = class_name[2:].strip()
    confidence = str(np.round(confidence_score * 100))[:-2]
    output_text = f"{currency} with confidence {confidence} percent"
    engine.say(output_text)
    engine.runAndWait()

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
