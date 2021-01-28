# python -m pip install plyer
from plyer import notification

def postureFix():
    notification.notify(
        title='Fix your posture',
        message='Idiot',
        app_icon='./code/kanye.ico',  # i want it to be kanye.ico'
        timeout=10,  # seconds
    )

# starter code taken from documentation
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
import numpy as np
import cv2
import pyvirtualcam
import pkgutil
import math





f = open('./haarcascade_frontalface_alt.xml', 'r')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# just face detection built into Open CV, must have it installed in C: drive
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
upper_body_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')
grey = 0
light = 0
tracking = False
good_posture = None
frozen = False
flip = False
filter_dict = {
    1: cv2.COLOR_BGR2RGB,
    2: cv2.COLOR_BGR2HSV,
    3: cv2.COLOR_BGR2Luv
}
text = ''

def brightnessControl(image, level):
    return cv2.convertScaleAbs(image, beta=level)

ret, frame = cap.read()

with pyvirtualcam.Camera(width=1280, height=720, fps=30) as cam:
    while(True):
        # Capture frame-by-frame
        if (not frozen):
            ret, frame = cap.read()
            # if (tracking):
            face = face_cascade.detectMultiScale(frame)
            print(face)
            for (ex,ey,ew,eh) in face:
                cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            if grey != 0:
                frame = cv2.cvtColor(frame, filter_dict[grey])
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            frame = brightnessControl(frame, light)
            # frame = cv2.medianBlur(frame, 5)
            if(flip):
                frame = cv2.flip(frame, -1)
        
        img = np.zeros((600,600,3), np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,'BCACam',(5,100), font, 3.5,(255, 0, 251),5,cv2.LINE_AA)
        cv2.putText(img,'By Tal Ledeniov and Remington Kim',(10,140), font, 0.75,(163, 0, 161),2,cv2.LINE_AA)
        cv2.putText(img,'Press \'f\' - freeze/unfreeze camera',(10,200), font, 0.75,(163, 0, 161),2,cv2.LINE_AA)
        cv2.putText(img,'Press \'m\' - change filter',(10,250), font, 0.75,(163, 0, 161),2,cv2.LINE_AA)
        cv2.putText(img,'Press \'l\' - flip the camera',(10,300), font, 0.75,(163, 0, 161),2,cv2.LINE_AA)
        cv2.putText(img,'Press \'q/z\' - brighten/darken camera',(10,350), font, 0.75,(163, 0, 161),2,cv2.LINE_AA)
        cv2.putText(img,'Press \'v\' - turn on/off tracking',(10,425), font, 0.75,(163, 0, 161),2,cv2.LINE_AA)
        cv2.putText(img,'Press \'c\' - calibrate to good posture',(10,475), font, 0.75,(163, 0, 161),2,cv2.LINE_AA)
        cv2.putText(img,'Press \'esc\' - close the program',(10,575), font, 0.75,(163, 0, 161),2,cv2.LINE_AA)

        cv2.imshow('image',img)

        # if(tracking):
        if(len(face)>0 and good_posture is not None):
            rmse = math.sqrt(sum([math.pow(face[0][i]-good_posture[i], 2) for i in range(len(face[0]))]))
            if rmse > 225:
                postureFix()
            print("The current RMSE is %.4f", rmse)   

        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):
            if grey < len(filter_dict):
                grey+=1
            else:
                grey = 0
        if k == ord('f'):
            frozen = not (frozen)
        if k == ord('c'):
            if(len(face)>0):
                good_posture = face[0]
        if k == ord('l'):
            flip = not flip
        if k == ord('q'):
            light+=10
        if k == ord('w'):
            light-=10
        if k == ord('v'):
            tracking = not (tracking)
        elif k == 27:
            break
        
        cam.send(frame)
        cam.sleep_until_next_frame()
        
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()