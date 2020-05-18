import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils import data
from torch.utils.data import Dataset
import os
from os import listdir
import PIL
from PIL import Image
import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


dataset_path = '../data/ucf_sports_actions/ucf_action'

all_classes = []
for class_name in listdir(dataset_path):
    if '.' in class_name:
        continue
    all_classes.append(class_name)
all_classes = sorted(all_classes)
print(all_classes)

index_to_classes = dict(enumerate(all_classes))
all_classes_to_index = {v: k for k, v in index_to_classes.items()}

class_to_imagepath = {}
for class_name in all_classes:
    class_to_imagepath[class_name] = []


for class_name in all_classes:
    class_path = dataset_path + '/' + class_name
    for group in listdir(class_path):
        if '.' in group:
            continue
        group_path = class_path + '/' + group
        for image_i in listdir(group_path):
            if '.jpg' not in image_i:
                continue
            image_path = group_path + '/' + image_i
            class_to_imagepath[class_name].append(image_path)


init_list_IDs = {}
init_labels = {}
count = 0
for class_name in all_classes:
    class_path = dataset_path + '/' + class_name
    for group in listdir(class_path):
        if '.' in group:
            continue
        group_path = class_path + '/' + group
        for image_i in listdir(group_path):
            if '.jpg' not in image_i:
                continue
            image_path = group_path + '/' + image_i
            
            init_list_IDs[count] = image_path
            init_labels[count] = all_classes_to_index[class_name]
            count += 1


class UCF_Sports_Dataset(data.Dataset):
#       '''Characterizes a dataset for PyTorch'''
    def __init__(self, list_IDs, labels):
        '''Initialization'''
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transforms.Compose(
                [transforms.Resize((250, 250)),
                    transforms.ToTensor(),
#                     transforms.CenterCrop(10),
                 
                 transforms.Normalize((0.5, 0.5, 0.5), 
                                      (0.5, 0.5, 0.5))])

    def __len__(self):
        '''Denotes the total number of samples'''
        return len(self.list_IDs)

    def __getitem__(self, index):
        '''Generates one sample of data'''
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        image = Image.open(image_path)
        image = self.transform(image)
        X = image
        y = self.labels[index]

        return X, y

ucf_dataset = UCF_Sports_Dataset(init_list_IDs, init_labels)

x,y = ucf_dataset.__getitem__(0)

data_loader = torch.utils.data.DataLoader(ucf_dataset,
                                          batch_size=4,
                                          shuffle=True,
                                         )
class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()
        # torch.Size([64, 3, 250, 250])
        # 3 input image channel (RGB), #6 output channels, 4x4 kernel 
        self.conv1 = nn.Conv2d(3, 6, kernel_size=(4,4), stride=1, 
                               padding=2, dilation=1, groups=1, 
                               bias=True, padding_mode='reflect')
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(3,3))
        self.conv3 = nn.Conv2d(16, 64, kernel_size=(3,3))
        self.conv4 = nn.Conv2d(64, 8, kernel_size=(4,4))
        self.fc1 = nn.Linear(128, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 13)
        
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 5)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.fc3(x)

        output = F.log_softmax(x, dim=1)
        return output
        
        
class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

alexNet = AlexNet(num_classes=13)
# Try different optimzers here [Adam, SGD, RMSprop]
optimizer = optim.RMSprop(alexNet.parameters(), lr=0.1)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Parameters
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 31

training_losses = []

# Generators
training_set = ucf_dataset
training_generator = data.DataLoader(training_set, **params)

# Loop over epochs
for epoch in range(max_epochs):
    # Training
    total_epoch_loss = 0
    for batch_idx, (batch_data, batch_labels) in enumerate(training_generator):
        
        output = alexNet(batch_data)
        target = batch_labels
        
        loss = F.nll_loss(output, target)   # Compute loss
        loss.backward()                     # Gradient computation
        optimizer.step()  
        total_epoch_loss += loss.item()
    
        if epoch%10 == 0 and batch_idx % 20 == 0:
            print('Train Epoch: {} \tLoss: {:.6f}'.format(
                epoch, total_epoch_loss))
    
    if epoch == 10:
        with open('../saved_models/alex_network1.pkl', 'wb') as f:
            torch.save(alexNet.state_dict(), f)
    if epoch == 20:
        with open('../saved_models/alex_network1.pkl', 'wb') as f:
            torch.save(alexNet.state_dict(), f)
    if epoch == 30:
        with open('../saved_models/alex_network1.pkl', 'wb') as f:
            torch.save(alexNet.state_dict(), f)
        
    training_losses.append(total_epoch_loss)
    


















