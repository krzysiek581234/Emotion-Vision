import cv2
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

from AugmentedDataset import AugmentedDataset
from CNN import CNN
from FER2013Dataset import FER2013Dataset
from HaarCascade import HaarCascade


# Defining plotting settings

plt.rcParams['figure.figsize'] = 14, 6

# paths to FER2013 CSV files
train_csv_file = './data/train/train.csv'
test_csv_file = './data/test/test.csv'

# Define the transformations to be applied
transform = transforms.Compose([
    transforms.ToPILImage(),        # Convert tensor to PIL Image
    transforms.ToTensor(),          # Convert PIL Image to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the image tensor
])

aug_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Create an instance of the FER2013 dataset
train_dataset = FER2013Dataset(train_csv_file, transform=transform)
test_dataset = FER2013Dataset(test_csv_file, transform=transform)

augmented_train_dataset = AugmentedDataset('augmented_images', aug_transform)

# Create a DataLoader to load the data in batches
batch_size = 128
train_loader = DataLoader(augmented_train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Plotting 25 images from the 1st batch
dataiter = iter(train_loader)
images, labels = next(dataiter)
plt.imshow(np.transpose(torchvision.utils.make_grid(
    images[:25], normalize=True, padding=1, nrow=5).numpy(), (1, 2, 0)))
plt.axis('off')
# plt.show()

# Selecting the appropriate training device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
model = CNN().to(device)

# haar face cascade classifier
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

# Define transformations to apply to the images
normalise_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((48, 48), antialias=True),                   # Resize to (48, 48)
    transforms.Normalize(mean=[0.5], std=[0.5])     # Normalize the tensor
])

# Function to preprocess face images
def preprocess_face(image):
    resized = normalise_transform(image)
    return resized

# HAAR CASCADE CLASSIFIER
haar = HaarCascade('./haarcascade_frontalface_default.xml')

# Defining the model hyper parameters
num_epochs = 30
learning_rate = 0.001
weight_decay = 0.01
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Training process begins
train_loss_list = []
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}:', end=' ')
    train_loss = 0

    # Iterating over the training dataset in batches
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        # Extracting images and target labels for the batch being iterated
        images = images.to(device)
        labels = labels.to(device)

        # Calculating the model output and the cross entropy loss
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Updating weights according to calculated loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Printing loss for each epoch
    train_loss_list.append(train_loss / len(train_loader))
    print(f"Training loss = {train_loss_list[-1]}")

# Plotting loss for all epochs
plt.plot(range(1, num_epochs + 1), train_loss_list)
plt.xlabel("Number of epochs")
plt.ylabel("Training loss")
plt.show()

torch.save(model.state_dict(), './model/model9.pth')

