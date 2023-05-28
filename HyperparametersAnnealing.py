import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import random
import math

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)


# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),

            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 7)
        )

    def forward(self, x):
        return self.model(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define hyperparameters and parameter space
initial_parameters = {
    'learning_rate': 0.001,
    'num_epochs': 25,
    'batch_size': 64
}

parameter_space = {
    'learning_rate': (0.0001, 0.01),
    'num_epochs': (20, 35),
    'batch_size': [32, 64, 128]
}

# Load the augmented images dataset
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

dataset = ImageFolder('/kaggle/input/fer2013augmented/augmented_images', transform=transform)
train_set, val_set = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), int(0.2 * len(dataset))])


# Define the cooling schedule
def cooling_schedule(initial_temperature, final_temperature, num_iterations):
    alpha = -math.log(final_temperature / initial_temperature)
    return lambda t: initial_temperature * math.exp(-alpha * t / num_iterations)


# Define the acceptance probability function
def acceptance_probability(old_loss, new_loss, temperature):
    if new_loss < old_loss:
        return 1.0
    else:
        return math.exp((old_loss - new_loss) / temperature)


# Initialize best parameters and loss
best_parameters = initial_parameters.copy()
best_loss = float('inf')

# Set initial temperature and number of iterations
initial_temperature = 10.0
final_temperature = 0.1
num_iterations = 30

# Create a cooling schedule and initialize current parameters
cooling_func = cooling_schedule(initial_temperature, final_temperature, num_iterations)
current_parameters = initial_parameters.copy()

# Simulated Annealing loop
for iteration in range(num_iterations):
    # Update the temperature
    temperature = cooling_func(iteration)

    print(f'ITERATION {iteration + 1}')
    print(f'Temperature: {temperature}')
    print('Current best parameters:')
    print(best_parameters)
    print('Current best loss:')
    print(best_loss)

    # Generate a new set of parameters
    new_parameters = current_parameters.copy()
    for param, value in current_parameters.items():
        if param == 'num_epochs':
            perturbation = random.randint(-2, 2)  # Perturbation range for num_epochs
            new_parameters[param] = value + perturbation
            new_parameters[param] = max(parameter_space[param][0],
                                        min(new_parameters[param], parameter_space[param][1]))
        elif param == 'learning_rate':
            perturbation = random.uniform(-0.0002, 0.0002)  # Perturbation range for learning_rate
            new_parameters[param] = round(value + perturbation, 4)
            new_parameters[param] = max(parameter_space[param][0],
                                        min(new_parameters[param], parameter_space[param][1]))
        else:
            new_parameters[param] = random.choice(parameter_space[param])

    print('New parameters:')
    print(new_parameters)

    # Train the model with the new parameters
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=new_parameters['learning_rate'])

    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=new_parameters['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=new_parameters['batch_size'], shuffle=True, num_workers=2)

    model.train()
    for epoch in range(new_parameters['num_epochs']):
        print(f'epoch {epoch + 1}')
        # Training
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Calculate the loss on the validation set
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    model.train()

    # Calculate acceptance probability and update current parameters
    accept_prob = acceptance_probability(best_loss, total_loss, temperature)
    if accept_prob > random.random():
        current_parameters = new_parameters.copy()
        best_parameters = new_parameters.copy()
        best_loss = total_loss

# Train the final model with the best parameters
final_model = CNN().to(device)
final_optimizer = optim.Adam(final_model.parameters(), lr=best_parameters['learning_rate'])

for epoch in range(best_parameters['num_epochs']):
    # Training
    final_model.train()
    for images, labels in train_loader:
        final_optimizer.zero_grad()
        outputs = final_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        final_optimizer.step()

torch.save(model.state_dict(), f'/kaggle/working/model_best.pth')
