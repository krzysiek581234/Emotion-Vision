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


TRAIN = False

# device name
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
log_dir = 'D:/Studia_Krzysiek/Semestr4/Projekty/Projekt_Sztuczna/Emotion-Vision/logs/TL'
testpath = 'D:/Studia_Krzysiek/Semestr4/Projekty/Projekt_Sztuczna/Emotion-Vision/dataset/OrgData/test_images'
val_csv_file = '../dataset/PublicTest.csv'
seed = 1444
torch.manual_seed(seed)
print("Seed "+ str(seed))


class readDataFCNN(Dataset):
    def __init__(self, rel, transforms):
        xy = pd.read_csv(rel)
        self.n_samples = len(xy)
        self.label = torch.from_numpy(np.array(xy.emotion.astype('float32')))
        self.features = torch.from_numpy(np.array(
            [np.array(xy.pixels[i].split(' ')).astype('float32') for i in range(self.n_samples)]))
        self.transforms = transforms

    def __getitem__(self, index):
        # aplikowanie normalizacji do features i zwrócenie par label, normalized_feature
        return self.label[index], self.transforms( torch.reshape(self.features[index], (1,48,48)) )

    def __len__(self):
        return self.n_samples



def main():
    batch_size = 128
    # Define transformations to apply to the input images
    transform = transforms.Compose([
        #transforms.CenterCrop((48,48)),
        transforms.Resize((48,48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])

    ])
    test_transform = transforms.Compose([
        #transforms.CenterCrop((48, 48)),

        transforms.Resize((48, 48)),
        transforms.Grayscale(3),
        transforms.Lambda(lambda image: torch.from_numpy(np.array(image).astype(np.float32)/255.0).permute((2, 0, 1))),
        #transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])

    ])

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logdir = os.path.join(log_dir, current_time + '_L_' + socket.gethostname())
    writer = SummaryWriter(logdir) # to see charts run:  tensorboard --logdir=runs

    # Load the augmented images dataset
    #dataset = ImageFolder('../dataset/augmented_images', transform=transform)

    #train_set, val_set = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])

    #val_set = readDataFCNN(val_csv_file,transform)

    test_set = ImageFolder(testpath, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Create data loaders
    #train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    #val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_classes = 7
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # freeze some layers - > early layers

    model = model.to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


    num_epochs = 30

    # Load for tests: resnet-transfer-30.pth
    #state_dict = torch.load('../savedModels/TL.pth', map_location=torch.device('cuda'))
    state_dict = torch.load('resnet-transfer-100.pth', map_location=torch.device('cuda'))
    model.load_state_dict(state_dict)
    model = model.to(device)
    val(model, test_loader, 100, writer, 'Test: ')
    print("Jeay")


    # atfer train:
    if writer is not None:
        writer.close()




def train(model, train_loader, val_loader, optimizer, criterion, num_epochs, writer,test_loader):
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

        # Validate:
        val(model, val_loader, epoch, writer)

        # Test:
        val(model, test_loader, epoch, writer, 'Test:')



def val(model, val_loader, epoch, writer, tag=''):
    with torch.no_grad():
        model.eval()
        n_correct = 0
        n_samples = 0
        labelPredictions = []
        labelActual = []
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            _, predictions = torch.max(outputs, 1)
            n_samples += labels.shape[0]
            n_correct += (predictions == labels).sum().item()
            labelPredictions += predictions.tolist()
            labelActual += labels.tolist()

        acc = 100.0 * n_correct / n_samples
        print(tag+f'Accuracy of the network on the val images: {acc} % in the Epoch {epoch + 1}')
        writer.add_scalar(tag+'Validation Accuracy', acc, epoch)

        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        fig, ax = plt.subplots(figsize=(10, 10))
        #    3.1 Calculate the confusion matrix when classifying the training data
        ax.title.set_text(tag+'Validation data confusion matrix:')
        cm = confusion_matrix(labelActual, labelPredictions)
        cmd = ConfusionMatrixDisplay(cm, display_labels=emotions)
        cmd.plot(ax=ax)
        if tag != '' and epoch+1 % 10 == 0:
            plt.show()
        fig.axes.append(ax)
        writer.add_figure(tag+"Val Confusion matrix", fig, epoch)

        # if epoch % 20 ==0 and epoch !=0:
        #    torch.save(model.state_dict(), f'resnet-transfer{epoch}.pth')




if __name__ == "__main__":
    main()