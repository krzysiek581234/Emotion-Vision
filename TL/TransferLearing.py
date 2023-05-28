import numpy as np
import pandas as pd
import os
import socket
from datetime import datetime
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import models
import torch.optim as optim
from tensorboardX import SummaryWriter
import pytorch_lightning as pl

#device name
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 128
# Define transformations to apply to the input images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
    # augmentatione !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
])

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
logdir = os.path.join('../../runs', current_time + '_' + socket.gethostname())
writer = SummaryWriter(logdir) # to see charts run:  tensorboard --logdir=runs

# Load the augmented images dataset
dataset = ImageFolder('../../dataset/augmented_images', transform=transform)
# Zbió walidacynju ba myć stały !!!!!!!!!!! - torch.manual_seed(...)
train_set, val_set = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])

# Create data loaders
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_classes = 7
model.fc = nn.Linear(model.fc.in_features, num_classes)
# freeze some layers - > early layers

model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


num_epochs =30

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total

    print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}')
    writer.add_scalar('Training Accuracy', train_accuracy, epoch)
    writer.add_scalar('Training Loss', train_loss, epoch)

    with torch.no_grad():
        model.eval()
        n_correct = 0
        n_samples = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            _, predictions = torch.max(outputs, 1)
            n_samples += labels.shape[0]
            n_correct += (predictions == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network on the test images: {acc} % in the Epoch {epoch + 1}')
        writer.add_scalar('Validation Accuracy', acc, epoch)
        if epoch % 20 ==0 and epoch !=0:
            torch.save(model.state_dict(), f'resnet-transfer{epoch}.pth')

if writer is not None:
    writer.close()

torch.save(model.state_dict(), 'resnet-transfer.pth')
