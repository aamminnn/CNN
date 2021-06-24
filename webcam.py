import cv2,os,urllib.request, dlib
import numpy as np
import time



eyes_detection = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

class VideoCamera(object):
	def __init__(self):
		self.video = cv2.VideoCapture(0)

	def __del__(self):
		self.video.release()

	def get_frame(self):
		while True:
			success, image = self.video.read()
			raw_img = image.copy() # for dataset
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			eyes_detected = eyes_detection.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
			for (x, y, w, h) in eyes_detected:
				cv2.rectangle(image, pt1=(x, y), pt2=(x + w, y + h), color=(0,255,0), thickness=2)
			frame_flip = cv2.flip(image,1)
			filename = '{}.jpg'.format(time.time())
			filepath1 = os.path.join('test_dataset/AuditoryConstruct', filename) # dataset folder
			filepath2 = os.path.join('test_dataset/AuditoryDigital', filename)
			filepath3 = os.path.join('test_dataset/AuditoryRecall', filename)
			filepath4 = os.path.join('test_dataset/Kinesthetics', filename)
			filepath5 = os.path.join('test_dataset/VisualConstruct', filename)
			filepath6 = os.path.join('test_dataset/VisualRecall', filename)
			file_array = [filepath1, filepath2, filepath3, filepath4, filepath5, filepath6]
			crop_img = cv2.flip(raw_img,1)[y:y+h, x-10:x+w+10] # crop eye image
			# cv2.imwrite(filepath, cv2.flip(raw_img,1))
			"""
			preprocessing to write files into different folder
			"""
			time.sleep(1)
			cv2.imwrite(filepath1, crop_img) # save to dataset
			cv2.imwrite(filepath2, crop_img)
			cv2.imwrite(filepath3, crop_img)
			cv2.imwrite(filepath4, crop_img)
			cv2.imwrite(filepath5, crop_img)
			cv2.imwrite(filepath6, crop_img)
			cv2.imshow("Demo", frame_flip)
			if cv2.waitKey(1) == 27:
				break

		self.video.release()
		cv2.destroyAllWindows()


		