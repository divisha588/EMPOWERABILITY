import cv2
import pyttsx3
import threading

# Initialize the pyttsx3 engine
engine = pyttsx3.init()

# Set properties (optional)
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 1)  # Volume (0.0 to 1.0)

# distance from camera to object(face) measured
Known_distance = 76.2

# width of face in the real world or Object Plane
Known_width = 14.3

# Colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# defining the fonts
fonts = cv2.FONT_HERSHEY_COMPLEX

# face detector object
face_detector = cv2.CascadeClassifier("Frontal.xml")


# focal length finder function
def Focal_Length_Finder(measured_distance, real_width, width_in_rf_image):
    # finding the focal length
    focal_length = (width_in_rf_image * measured_distance) / real_width
    return focal_length


# distance estimation function
def Distance_finder(Focal_Length, real_face_width, face_width):
    distance = (real_face_width * Focal_Length) / face_width
    return distance


def face_data(image):
    face_width = 0

    # converting color image ot gray scale image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detecting face in the image
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)

    # looping through the faces detect in the image
    # getting coordinates x, y , width and height
    for (x, y, h, w) in faces:
        # draw the rectangle on the face
        cv2.rectangle(image, (x, y), (x + w, y + h), GREEN, 2)

        # getting face width in the pixels
        face_width = w

        # return the face width in pixel
    return face_width


# reading reference_image from directory
ref_image_face = cv2.imread("Ref_image_face.jpg")

# find the face width(pixels) in the reference_image
ref_image_face_width = face_data(ref_image_face)

# get the focal by calling "Focal_Length_Finder"
Focal_length_found = Focal_Length_Finder(Known_distance, Known_width, ref_image_face_width)

print(Focal_length_found)

# initialize the camera object so that we can get frame from it
cap = cv2.VideoCapture(0)

# Flag to control the while loop
running = True


# function to speak the text
def speak_text(text):
    if not engine._inLoop:
        engine.say(text)
        engine.runAndWait()


# looping through frame, incoming from camera/video
while running:

    # reading the frame from camera
    _, frame = cap.read()

    # calling face_data function to find
    # the width of face(pixels) in the frame
    face_width_in_frame = face_data(frame)

    # check if the face is zero then not
    # find the distance
    if face_width_in_frame != 0:
        # finding the distance by calling function
        # Distance finder function need
        # these arguments the Focal_Length,
        # Known_width(centimeters),
        # and Known_distance(centimeters)
        Distance = Distance_finder(
            Focal_length_found, Known_width, face_width_in_frame)

        Distance = round(Distance, 0)

        text = "Man detected at a distance of " + str(Distance) + " centimetres "

        # Speak the text in a separate thread
        thread = threading.Thread(target=speak_text, args=(text,))
        thread.start()

        # draw line as background of text
        cv2.line(frame, (30, 30), (230, 30), RED, 32)
        cv2.line(frame, (30, 30), (230, 30), BLACK, 28)

        # Drawing Text on the screen
        cv2.putText(frame, f"Distance: {round(Distance, 2)} CM", (30, 35), fonts, 0.6, GREEN, 2)

    # show the frame on the screen
    cv2.imshow("frame", frame)

    # quit the program if you press 'q' on keyboard
    key = cv2.waitKey(1)
    if key == ord("q"):
        running = False

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Exit the program
exit()
