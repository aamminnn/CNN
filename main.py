import PyQt5.QtWidgets as qtw
import PyQt5.QtGui as qtg
from PyQt5.QtCore import pyqtSlot, QSize, QRect
from webcam import VideoCamera
from webcam import *
import cv2
from CNN import *

gaze = VideoCamera()
eyes_detection = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
# input_data = []
stimulus = [
			"What is the colour of the shirt you wore yesterday?",
			"What would your room look like if it were painted yellow with big purple circles?",
			"What does your best friend's voice sound like?",
			"What will your voice sound like in 10 years?",
			"What is something you continually tell yourself?",
			"What does it feel like to walk barefoot on a cool sandy beach?"
			]

class MainWindow(qtw.QWidget):
	def __init__(self):
		super().__init__()

		# Add a title
		self.setWindowTitle("Learning style Identifier")

		# Set Vertical layout
		self.setLayout(qtw.QVBoxLayout())

		# Create a label
		my_label = qtw.QLabel("Learning Style Identification") 
		self.layout().addWidget(my_label)

		# # Text Edit
		# display_result = qtw.QTextEdit()
		# self.layout().addWidget(display_result)

		# Change font size of label
		my_label.setFont(qtg.QFont('Helvetica', 20))
		self.layout().addWidget(my_label)

		# Create an entry box
		# my_entry = qtw.QLineEdit()
		# my_entry.setObjectName("name_field")
		# my_entry.setText("")
		# self.layout().addWidget(my_entry)

		# # Create train button
		# train = qtw.QPushButton("Train the model",
		# 	clicked = lambda: train())
		# self.layout().addWidget(train)

		# Create start button
		my_button = qtw.QPushButton("Start Application",
			clicked = lambda: run_app())
		self.layout().addWidget(my_button)

		# Create result button
		my_button2 = qtw.QPushButton("ViewResult",
			clicked = lambda: result())
		self.layout().addWidget(my_button2)

		display = qtw.QTextEdit()
		self.layout().addWidget(display)

		
		# self.textbox = qtw.QTextEdit(self)
		# self.textbox.move(50., 210)
		# self.textbox.resize(540, 200)
		# self.textbox.setReadOnly(True)
		

		# self.output = qtw.QLabel(" ", self)
		# self.output.setWordWrap(True)
		# self.output.setGeometry(QtCore.QRect(self.output.x(), self.output.y(), self.output.width()+150, self.output.height()))
		# self.output.move(10,120)

		# Show the app
		self.show()

		# def train():
		# 	train_model(trainpath)

		# def press_it():
		# 	my_label.setText(f'Hello {my_entry.text()}')
		# 	# Clear the entry box
		# 	my_entry.setText("")

		def run_app():
			# display_stimulus = qtw.QTextEdit()
			# self.layout().addWidget(display_stimulus)
			for i in range(len(stimulus)):
				display.append(str(stimulus[i]))
			gaze.get_frame()


			# while True:
			# 	success, image = gaze.video.read()
			# 	raw_img = image.copy() # for dataset
			# 	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			# 	eyes_detected = eyes_detection.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
			# 	for (x, y, w, h) in eyes_detected:
			# 		cv2.rectangle(image, pt1=(x, y), pt2=(x + w, y + h), color=(0,255,0), thickness=2)
			# 	frame_flip = cv2.flip(image,1)
			# 	filename = '{}.jpg'.format(time.time())
			# 	filepath = os.path.join('user_dataset', filename) # dataset folder
			# 	crop_img = cv2.flip(raw_img,1)[y:y+h, x-10:x+w+10] # crop eye image
			# 	"""
			# 	pre processing take frame each 5s
			# 	"""
			# 	# cv2.imwrite(filepath, crop_img) # save to dataset
			# 	cv2.imshow("Demo", frame_flip)

			# 	display_stimulus = qtw.QTextEdit()
			# 	self.layout().addWidget(display_stimulus)
			# 	display_stimulus.append(stimulus[0])

			# 	if cv2.waitKey(1) == 27:
			# 		break



			# gaze.get_frame()
			# display_stimulus = qtw.QTextEdit()
			# self.layout().addWidget(display_stimulus)
			# display_stimulus.append(stimulus[0])

		def result():
			display.setText("")
			print("result")
			input_data = load_input_data(inputpath)
			input_data_trim = []
			for i in range(30):
				input_data_trim += input_data

			# print(input_data_trim)

			model = Net().to(device)
			model.load_state_dict(torch.load(model_path))
			with torch.no_grad():
				n_correct = 0
				n_samples = 0
				n_class_correct = [0 for i in range(6)]
				n_class_samples = [0 for i in range(6)]
				for images, labels in input_data_trim:
					images = images.to(device)
					labels = labels.to(device)
					outputs = model(images)
					_, predicted = torch.max(outputs,1)
					n_samples += labels.size(0)
					n_correct += (predicted == labels).sum().item()

					for i in range(batch_size):
						label = labels[i]
						pred = predicted[i]
						if (label == pred):
							n_class_correct[label] += 1
						n_class_samples[label] += 1

				acc_network = 100.0 * n_correct/n_samples
				# my_label.setText(str(f'Accuracy of network: {acc_network} %'))
				display.append(str(f'Accuracy of network: {acc_network} %'))

				for i in range(6):
					acc_classes = 100.0 * n_class_correct[i] / n_class_samples[i]
					display.append(str(f'Accuracy of {classes[i]}: {acc_classes} %'))


			# Text Edit
			# display_result = qtw.QTextEdit()
			# self.layout().addWidget(display_result)
			# currently testing output for testdata, so use testpath
			# print(input_data.shape)
			# test_accuracy(test_data)
			# result1, result2 = test_accuracy(test_data)
			# my_label.setText(str(result1))
			# my_label2.setText(str(result2))
			# self.textbox.setText(str(result1))
			# result = ""
			# result = result1
			# self.output.setText(result)



app = qtw.QApplication([])
mw = MainWindow()

# Run the App
app.exec_()