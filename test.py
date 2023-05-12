import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import cv2
import numpy as np

from CNN import CNN
from FER2013Dataset import FER2013Dataset

test_acc = 0
model = CNN()  # Create an instance of the model
model.load_state_dict(torch.load('./model/model3.pth'))
model.eval()

test_csv_file = './data/test/test.csv'

# Define the transformations to be applied
transform = transforms.Compose([
    transforms.ToPILImage(),        # Convert tensor to PIL Image
    transforms.ToTensor(),          # Convert PIL Image to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the image tensor
])

test_dataset = FER2013Dataset(test_csv_file, transform=transform)
batch_size = 128
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

with torch.no_grad():
    # Iterating over the test dataset in batches
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        y_true = labels.to(device)

        # haar cascade classifier image preprocessing
        transformed_images = []
        for image in images:
            transformed_images.append(preprocess_face(image))
            # numpy_image = image.numpy()
            # if numpy_image.shape[0] == 1:
            #    numpy_image = np.squeeze(numpy_image, axis=0)
            # else:
            #    numpy_image = np.transpose(numpy_image, (1, 2, 0))
            # cv2.imshow('Image', numpy_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        transformed_images = torch.stack(transformed_images)

        # Calculating outputs for the batch being iterated
        outputs = model(transformed_images)

        # Calculated prediction labels from models
        _, y_pred = torch.max(outputs.data, 1)

        # Comparing predicted and true labels
        test_acc += (y_pred == y_true).sum().item()

    print(f"Test set accuracy = {100 * test_acc / len(test_dataset)} %")
