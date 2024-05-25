import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
counter = 0
folder = r"C:\Users\divisha\PycharmProjects\SignLanguageDetection\Data\Happy"

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Ensure cropped region is within the bounds of the image
        if y - offset >= 0 and x - offset >= 0 and y + h + offset <= img.shape[0] and x + w + offset <= img.shape[1]:
            imgCrop = img[y - offset: y + h + offset, x - offset: x + w + offset]

            # Ensure cropped image is not empty
            if not imgCrop.size:
                print("Empty cropped image!")
            else:
                imgCropShape = imgCrop.shape
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap: wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap: hCal + hGap, :] = imgResize

                cv2.imshow('imageCrop', imgCrop)
                cv2.imshow('imageWhite', imgWhite)

                key = cv2.waitKey(1)
                if key == ord('s'):
                    counter += 1
                    cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
                    print(counter)
        else:
            print("Cropped region is out of bounds.")

    cv2.imshow("Image", img)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
