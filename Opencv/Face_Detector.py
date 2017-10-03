
# coding: utf-8

# In[ ]:

import cv2
import sys
import os
import errno
from glob import glob
def Detector():
	faceCascade = cv2.CascadeClassifier("C:/Users/Downloads/OpenCV/haarcascade_frontalface_default.xml")
	eye_left_cascade = cv2.CascadeClassifier("C:/Users/Downloads/OpenCV/haarcascade_lefteye_2splits.xml")
	eye_right_cascade = cv2.CascadeClassifier("C:/Users/Downloads/OpenCV/haarcascade_righteye_2splits.xml")
	smile_cascade = cv2.CascadeClassifier("C:/Users/Downloads/OpenCV/haarcascade_smile.xml")


	webcam = cv2.VideoCapture(0)

	while (True):
		ret, frame = webcam.read()
		if ret == True:
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			faces = faceCascade.detectMultiScale(
				gray,
				scaleFactor = 1.07,
				minNeighbors = 8,
				minSize = (33,33),
				flags = cv2.cv.CV_HAAR_SCALE_IMAGE
				)
		
			for (x,y,w,h) in faces:
				cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

				gray_ = gray[y:y+h, x:x+w]
				roi_color = frame[y:y+h, x:x+w] 

				reyes = eye_right_cascade.detectMultiScale(gray_, 1.06, 5)

				for (xx,yy,ww,hh) in reyes:
					cv2.rectangle(roi_color, (xx,yy), (xx+ww,yy+hh), (0,255,0), 2)           
				leyes = eye_left_cascade.detectMultiScale(gray_, 1.06, 5)

				for (xx,yy,ww,hh) in leyes:
					cv2.rectangle(roi_color, (xx,yy), (xx+ww,yy+hh), (255,0,0), 2)
           
				sms = smile_cascade.detectMultiScale(gray_ , 1.4 , 43 )
				for (xx,yy,ww,hh) in sms:
					cv2.rectangle(roi_color, (xx,yy), (xx+ww,yy+hh), (0 , 0 , 255) , 2)
                
			cv2.imshow("webcam", frame)
			if cv2.waitKey(300) & 0xFF == ord("q"):
				break
            

	#webcam.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
    Detector()

