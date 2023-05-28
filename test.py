import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from CNN import CNN
from TestMetrics import TestMetrics

test_acc = 0
model = CNN()
model.load_state_dict(torch.load('./model/model_final.pth', map_location=torch.device('cpu')))
model.eval()

# Define transformations to apply to the input images
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load the augmented images dataset
test_dataset = ImageFolder('./test_images', transform=transform)

# Create data loader
batch_size = 128
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    predicted_labels = []
    true_labels = []
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predicted_labels.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Test set accuracy = {accuracy:.2f}%')

    metrics = TestMetrics(true_labels, predicted_labels)
    metrics.print_metrics()
