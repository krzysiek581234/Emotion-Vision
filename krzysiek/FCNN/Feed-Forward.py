import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from NeuralNet import NeuralNet
from krzysiek.FCNN.jcoopdateread import FER2013Vector
from readDataFCNN import readDataFCNN
from tensorboardX import SummaryWriter
#device name
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyper
input_size =  2304 #48x48
num_classes = 7
num_epochs = 100
batch_size = 32#100
learning_rate = 0.001
Leaning_csv = '../../dataset/LearningTest.csv'
test_csv_file = '../../dataset/PublicTest.csv'

writer = SummaryWriter()

train_dataset = readDataFCNN(Leaning_csv)
test_dataset = readDataFCNN(test_csv_file)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

examples = iter(train_loader)
labels, features  = examples.__next__()

# TESTS IMAGES SHOW
# print(features.shape, labels.shape)
# matrix_tensor = features[0].view(48, 48)
# print(matrix_tensor.shape)
#
# for i in range(6):
#     plt.subplot(2,3,i+1)
#     matrix_tensor = features[i].view(48, 48)
#     plt.imshow(matrix_tensor, cmap='gray')
# plt.axis('off')
# plt.show()
#model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1,num_classes=7)
model = NeuralNet(input_size, num_classes)
model.to(device)
# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def validate(model, test_loader, epoch):
    with torch.no_grad():
        model.eval()
        n_correct = 0
        n_samples = 0
        for labels, images in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            _, predictions = torch.max(outputs, 1)
            n_samples += labels.shape[0]
            n_correct += (predictions == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network on the test images: {acc} % in the Epoch {epoch+1}')
        writer.add_scalar('Validation Accuracy', acc, epoch)


    #traning
n_total = len(train_loader)
for epoch in range(num_epochs):
    model.train()
    n_correct = 0
    n_samples = 0
    for i, (labels, images) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # forward
        outputs = model(images)
        loss = criterion(outputs, labels.type(torch.int64))

        #backwards
        loss.backward()
        optimizer.step()

        #calculate traning accuracy
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

        if(i+1) % 100 == 0:
            print(f"epoch {epoch +1} / { num_epochs}, step {i+1}/{n_total}, loss: {loss}")

    acc = 100.0 * n_correct / n_samples

    print(f"Finish epoch {epoch + 1} of traning with tracc {acc} %")
    writer.add_scalar('Training Accuracy', acc, epoch)
    #test
    if epoch % 20 == 0 and epoch != 0:
        validate(model, test_loader,epoch)

#torch.save(model,'feed-Forward.pth')
torch.save(model.state_dict(), 'feed-Forward.pth')
validate(model,test_loader,99)


