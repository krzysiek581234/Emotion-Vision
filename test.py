import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import cv2
import numpy as np
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix

from CNN import CNN
from FER2013Dataset import FER2013Dataset

test_acc = 0
model = CNN()  # Create an instance of the model
model.load_state_dict(torch.load('./model/model_302.pth', map_location=torch.device('cpu')))
model.eval()

test_csv_file = './data/test/test.csv'

# Define the transformations to be applied
transform2 = transforms.Compose([
    transforms.ToPILImage(),        # Convert tensor to PIL Image
    transforms.ToTensor(),          # Convert PIL Image to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the image tensor
])

# Define transformations to apply to the input images
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load the augmented images dataset
test_dataset = ImageFolder('./test_images', transform=transform)

# Create data loaders
batch_size = 128
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# test_dataset = FER2013Dataset(test_csv_file, transform=transform)

# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# haar face cascade classifier
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

# Define transformations to apply to the images
normalise_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((48, 48), antialias=True),                   # Resize to (48, 48)
    transforms.Normalize(mean=[0.5], std=[0.5])     # Normalize the tensor
])

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    predicted_labels = []
    true_labels = []
    for images, labels in test_loader:
        images = images.to(device)
        print(images)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predicted_labels.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Test set accuracy = {accuracy:.2f}%')

    # Create confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    print('Confusion Matrix:')
    print(cm)
