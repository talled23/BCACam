# starter code taken from documentation
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
# just face detection built into Open CV, must have it installed in C: drive
upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+ 'haarcascade_frontalface_alt.xml')
# upper_body_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    # apply filters here probably
    # changing color hue can be done here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # the detection
    upper_body = upper_body_cascade.detectMultiScale(gray, 1.3, 5)

    # prints data to console for now
    print(upper_body)


    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()