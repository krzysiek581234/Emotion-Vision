import torch
import matplotlib.pyplot as plt
from torchvision.transforms import transforms

from CNN import CNN
from FER2013Dataset import FER2013Dataset

aug_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Create an instance of the FER2013 dataset
dataset = FER2013Dataset(aug_transform)

train_loader = dataset.load_train_dataset()
test_loader = dataset.load_test_dataset()

# Selecting the appropriate training device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CNN().to(device)

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

torch.save(model.state_dict(), './model/model0.pth')

