
# coding: utf-8

# In[ ]:

import cv2
import numpy
import sys
import os
import errno
from glob import glob

def Recognizer():
	size = 4
	faceCascade = cv2.CascadeClassifier("C:/Users/Downloads/OpenCV/haarcascade_frontalface_default.xml")
	eye_left_cascade = cv2.CascadeClassifier("C:/Users/Downloads/OpenCV/haarcascade_lefteye_2splits.xml")
	eye_right_cascade = cv2.CascadeClassifier("C:/Users/Downloads/OpenCV/haarcascade_righteye_2splits.xml")
	smile_cascade = cv2.CascadeClassifier("C:/Users/Downloads/OpenCV/haarcascade_smile.xml")
	m_directory = "C:/Users/Downloads/OpenCV/faces"
    
	imgs, lb_, fname = [], [], {}
	for __, dirs, __ in os.walk(m_directory):
		for id,subdir in enumerate(dirs):
			fname[id] = subdir
			s_path = os.path.join(m_directory, subdir)
			for nfile in os.listdir(s_path):
				nnam, ex_ = os.path.splitext(nfile)
				EX = '.png .jpg .jpeg .gif .pgm'.split()
				if ex_.lower() not in EX:
					continue
				path = s_path + '/' + nfile
				lb = id
				imgs.append(cv2.imread(path, 0))
				lb_.append(int(lb))
			id += 1

	(imgs, lb_) = [numpy.array(lists) for lists in [imgs, lb_]]
	model = cv2.createFisherFaceRecognizer()
	model.train(imgs, lb_)
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
				resizer = cv2.resize(gray_, (100,100))
				pre = model.predict(resizer)
				if pre[1]<500:
					cv2.putText(frame,'%s - %.0f' % (fname[pre[0]],pre[1]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
				else:
					cv2.putText(frame,'Not Recognized',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))

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
    Recognizer()

