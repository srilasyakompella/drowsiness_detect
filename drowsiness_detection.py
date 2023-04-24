#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy
import dlib
import imutils
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
#import argparse
import time
from pygame import mixer

mixer.init()
sound = mixer.Sound('alarm.wav')


# In[2]:


def euclidean_dist(ptA, ptB):
    
    return np.linalg.norm(ptA - ptB)

def eye_aspect_ratio(eye):
    A = euclidean_dist(eye[1], eye[5])
    B = euclidean_dist(eye[2], eye[4])
    # eye landmark (x, y)-coordinates
    C = euclidean_dist(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    # return the eye aspect ratio
    return ear


# In[3]:


EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 16
MOUTH_AR_THRESH = 0.82
# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off

COUNTER = 0
ALARM_ON = False
(mStart, mEnd) = (49, 68)

detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

print("[INFO] loading facial landmark predictor...")

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


# In[4]:



from scipy.spatial import distance as dist

def mouth_aspect_ratio(mouth):
    # compute the euclidean distances between the two sets of
    # vertical mouth landmarks (x, y)-coordinates
    A = dist.euclidean(mouth[2], mouth[10])  # 51, 59
    B = dist.euclidean(mouth[4], mouth[8])  # 53, 57

    # compute the euclidean distance between the horizontal
    # mouth landmark (x, y)-coordinates
    C = dist.euclidean(mouth[0], mouth[6])  # 49, 55

    # compute the mouth aspect ratio
    mar = (A + B) / (2.0 * C)

    # return the mouth aspect ratio
    return mar


# In[5]:


#cap = cv2.VideoCapture('/home/ravindra/Downloads/WhatsApp Video 2023-04-10 at 16.10.55.mp4')
cap = cv2.VideoCapture(1)

# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video stream or file")

while(cap.isOpened()):
    
    ret, frame = cap.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale
        # channels)
        
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect faces in the grayscale frame
        rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in rects:
            rect = dlib.rectangle(int(x), int(y), int(x + w),
                int(y + h))
            
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            mouth = shape[mStart:mEnd]

            mouthMAR = mouth_aspect_ratio(mouth)
            mar = mouthMAR
            # compute the convex hull for the mouth, then
            # visualize the mouth
            mouthHull = cv2.convexHull(mouth)

            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
            cv2.putText(frame, "MAR: {:.2f}".format(mar), (650, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if mar > MOUTH_AR_THRESH:
                try:
                    sound.play()

                except:  # isplaying = False
                    pass
                cv2.putText(frame, "Yawning!", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)            
            if ear < EYE_AR_THRESH:
                COUNTER += 1
#                 print(COUNTER)
                # if the eyes were closed for a sufficient number of
                # frames, then sound the alarm
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    # if the alarm is not on, turn it on
                    if not ALARM_ON:
                        ALARM_ON = True
                        # check to see if the TrafficHat buzzer should
                        # be sounded
                        #if args["alarm"] > 0:
                            #th.buzzer.blink(0.1, 0.1, 10,background=True)
                    # draw an alarm on the frame
                        try:
                            sound.play()

                        except:  # isplaying = False
                            pass
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0
                ALARM_ON = False
                cv2.putText(frame, "EAR: {:.3f}".format(ear), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # show the frame
        cv2.imshow("Frame", frame)
        
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

      # Break the loop
    else:
        break
    
    
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




