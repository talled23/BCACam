# starter code taken from documentation
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
import numpy as np
import cv2
f = open('./haarcascade_frontalface_alt.xml', 'r')
cap = cv2.VideoCapture(0)
# just face detection built into Open CV, must have it installed in C: drive
upper_body_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
# upper_body_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
grey = 0
frozen = False

while(True):
    # Capture frame-by-frame
    
    ret, frame = cap.read()

    if grey % 2 == 0:
        color = cv2.COLOR_BGR2GRAY
    else:
        color = cv2.COLOR_BGR2HSV_FULL

    # Our operations on the frame come here
    # apply filters here probably
    # changing color hue can be done here
    gray = cv2.cvtColor(frame, color)

    
    # the detection
    upper_body = upper_body_cascade.detectMultiScale(gray, 1.3, 5)

    # prints data to console for now
    print(upper_body)


    # Display the resulting frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(gray,'HiRemi',(10,300), font, 4,(255,255,255),2,cv2.LINE_AA)
    cv2.imshow('frame',gray)
     
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        grey += 1
    if k == ord('f'):
        frozen = not (frozen)
    elif k == 27:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()