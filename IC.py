#!/usr/bin/env python
# coding: utf-8

# Developing an AI application

# Imports here
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt


# Load the data


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'



#  Define your transforms for the training, validation, and testing sets

training_transform = transforms.Compose([transforms.RandomRotation(30), transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

testing_transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

valid_transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                                          
                                      
# Load the datasets with ImageFolder

train_data = datasets.ImageFolder(train_dir, transform = training_transform)
test_data = datasets.ImageFolder(test_dir, transform = testing_transform)
valid_data = datasets.ImageFolder(valid_dir, transform = valid_transform)
                                          
                                          
# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)



dataiter = iter(trainloader)
inputs, labels = dataiter.next()

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


# Use GPU if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.vgg16(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
    
model.classifier = nn.Sequential(nn.Linear(25088, 6572), nn.ReLU(), nn.Dropout(0.2), 
                                 nn.Linear(6572, 1568), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(1568, 102), nn.ReLU(), nn.Dropout(0.2),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

model.to(device);


epochs = 5
steps = 0
running_loss = 0
print_every = 60
print('starting')

for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:   
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                      
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()           

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(validloader):.3f}.. "
                  f"Test accuracy: {accuracy.double()/len(validloader):.3f}")
            
            running_loss=0
            model.train()
            


plt.plot(running_loss, label='Training loss')
plt.plot(test_loss, label='Validation loss')
plt.legend(frameon=False)
