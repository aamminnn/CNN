import numpy as np
import torch
from torch import nn
from torch import optim
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plts
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classes = ('AuditoryConstruct', 'AuditoryDigital', 'AuditoryRecall', 'Kinesthetics', 'VisualConstruct', 'VisualRecall')
batch_size = 5 # train use 50
num_epochs = 4
# PATH = 'D:\\SEM 7\\FYP\\Trydataset\\model.pth'
filename = 'model.pth'
filepath = os.path.join("raw_dataset", filename)
trainpath = "raw_dataset/traindata"
testpath = "raw_dataset/testdata"
inputpath = "test_dataset/"
model_path = "raw_dataset/model.pth"

def load_train_data(trainpath):
	transform = transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])
	trainset = torchvision.datasets.ImageFolder(trainpath, transform = transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size)
	return trainloader

def load_test_data(testpath):
	transform = transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])
	testset = torchvision.datasets.ImageFolder(testpath, transform = transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size)
	return testloader

def load_input_data(inputpath):
	transform = transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])
	inputset = torchvision.datasets.ImageFolder(inputpath, transform = transform)
	inputloader = torch.utils.data.DataLoader(inputset, batch_size = batch_size)
	return inputloader

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16*5*5, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, 6)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16*5*5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# train the model
def train_model(trainloader):
	n_total_steps = len(trainloader)
	for epoch in range(num_epochs):
		running_loss = 0.0
		for i, (images, labels) in enumerate(trainloader):
			images = images.to(device)
			labels = labels.to(device)

			# forward + backward + optimize
			outputs = model(images)
			loss = criterion(outputs, labels)

			# zero the parameter gradients
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if (i+1) & 1200 == 0:
				print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

	torch.save(model.state_dict(), filepath)

# function to test accuracy on train dataset, not being used on gui
def train_accuracy(trainloader):
	# load in model
	model = Net().to(device)
	model.load_state_dict(torch.load(model_path))
	with torch.no_grad():
		n_correct = 0
		n_samples = 0
		n_class_correct = [0 for i in range(6)]
		n_class_samples = [0 for i in range(6)]
		for images, labels in trainloader:
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
		print(f'Accuracy of the network: {acc_network} %')

		for i in range(6):
			acc_classes = 100.0 * n_class_correct[i] / n_class_samples[i]
			print(f'Accuracy of {classes[i]}: {acc_classes} %')

	return acc_network, acc_classes

# function to test accuracy on test dataset, not being used on gui
def test_accuracy(testloader):
	# load in model
	model = Net().to(device)
	model.load_state_dict(torch.load(model_path))
	with torch.no_grad():
		n_correct = 0
		n_samples = 0
		n_class_correct = [0 for i in range(6)]
		n_class_samples = [0 for i in range(6)]
		for images, labels in testloader:
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
		print(f'Accuracy of the network: {acc_network} %')

		for i in range(6):
			acc_classes = 100.0 * n_class_correct[i] / n_class_samples[i]
			print(f'Accuracy of {classes[i]}: {acc_classes} %')

	return acc_network, acc_classes

# function to test accuracy on inputs data
def input_accuracy(inputloader):
	# load in model
	model = Net().to(device)
	model.load_state_dict(torch.load(model_path))
	with torch.no_grad():
		n_correct = 0
		n_samples = 0
		n_class_correct = [0 for i in range(6)]
		n_class_samples = [0 for i in range(6)]
		for images, labels in inputloader:
			images = images.to(device)
			labels = labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs,1)
			n_samples += labels.size(0)
			n_correct += (predicted == labels).sum().item()
			# print(batch_size)
			# print(labels)

			for i in range(batch_size):
				label = labels[i]
				pred = predicted[i]
				if (label == pred):
					n_class_correct[label] += 1
				n_class_samples[label] += 1

		acc_network = 100.0 * n_correct/n_samples
		print(f'Accuracy of the network: {acc_network} %')

		for i in range(6):
			acc_classes = 100.0 * n_class_correct[i] / n_class_samples[i]
			print(f'Accuracy of {classes[i]}: {acc_classes} %')

	return acc_network, acc_classes




"""
dir = path contain input images( D:\\...) #raw
input = load_input_data(dir)
result = model_accuracy(input)
print(result)

"""