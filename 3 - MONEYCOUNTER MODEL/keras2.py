from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

try:
    # Load the model
    model = load_model(r"C:\Users\divisha\Downloads\money counter\converted_keras (1)\keras_model.h5", compile=False)

    # Load the labels
    class_names = open(r"C:\Users\divisha\Downloads\money counter\converted_keras (1)\labels.txt", "r").readlines()

    # CAMERA can be 0 or 1 based on the default camera of your computer
    camera = cv2.VideoCapture(0)

    # Set the desired window size
    window_width = 1280  # Width of the windowx
    window_height = 720  # Height of the window

    while True:
        # Grab the web camera's image.
        ret, image = camera.read()

        # Check if the camera is opened successfully
        if not ret:
            print("Failed to grab the image from the camera.")
            break

        # Resize the captured image to the desired window size
        resized_image = cv2.resize(image, (224, 224))

        # Make the image a numpy array and reshape it to the model's input shape.
        image_for_prediction = np.asarray(resized_image, dtype=np.float32).reshape(1, 224, 224, 3)

        # Normalize the image array
        image_for_prediction = (image_for_prediction / 127.5) - 1

        # Predicts the model
        prediction = model.predict(image_for_prediction)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]

        # Display the detected class name on the camera screen
        cv2.putText(image, "Class: " + class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the image in a window
        cv2.imshow("Webcam Image", image)

        # Print prediction and confidence score
        print("Class:", class_name, end="")
        print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

        # Listen to the keyboard for presses.
        keyboard_input = cv2.waitKey(1)

        # 27 is the ASCII for the esc key on your keyboard.
        if keyboard_input == 27:
            break

    camera.release()
    cv2.destroyAllWindows()

except Exception as e:
    print("Exception encountered:", e)
