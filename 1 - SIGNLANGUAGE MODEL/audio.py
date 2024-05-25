import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3
import threading

# Initialize the TTS engine
engine = pyttsx3.init()

# Function to speak the text
def speak_text(text):
    engine.say(text)
    engine.runAndWait()

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier(r"C:\Users\divisha\PycharmProjects\SignLanguageDetection\converted_keras\keras_model.h5", r"C:\Users\divisha\PycharmProjects\SignLanguageDetection\converted_keras\labels.txt")

# Manually compile the model after loading
classifier.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

offset = 20
imgSize = 300
counter = 0

labels = ["Happy", "Hello", "No", "Please", "Sad", "Thank you", "Washroom", "Yes"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape

        if imgCropShape[0] <= 0 or imgCropShape[1] <= 0:
            print("Invalid crop dimensions:", imgCropShape)
            continue  # Skip processing this frame

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            if wCal <= 0 or imgSize <= 0:
                print("Invalid resize dimensions:", wCal, imgSize)
                continue  # Skip processing this frame
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap: wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            if hCal <= 0 or imgSize <= 0:
                print("Invalid resize dimensions:", hCal, imgSize)
                continue  # Skip processing this frame
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0),
                      cv2.FILLED)

        label_text = labels[index]

        cv2.putText(imgOutput, label_text, (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

        # Speak the detected label in a separate thread
        thread = threading.Thread(target=speak_text, args=(label_text,))
        thread.start()

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()