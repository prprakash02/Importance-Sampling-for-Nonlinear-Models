import torch, datetime
import os, random, numpy as np
import torch.nn as nn
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from ImageModel import SimpleNNBinary
import torch.nn.functional as F
import matplotlib.pyplot as plt

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set the seed for reproducibility
set_seed(42)

image_size = 10
IMG_DIR = ''
SAVE_PATH = ''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# Define the transformation to resize images and convert to tensor
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Resize((image_size, image_size))
])

# Load the dataset
train_dataset = torchvision.datasets.ImageFolder(root=f'{IMG_DIR}/train', transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Define the neural network model
num_epochs = 1000
model = SimpleNNBinary()
model = model.to(device)
time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
print(time, flush=True)

def custom_function(x):
    return torch.sigmoid(x)

# Define the loss function and optimizer
# criterion = nn.BCELoss()
criterion = nn.BCEWithLogitsLoss()

# optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)

def plot(train_losses, train_accuracies, train_accuracy, best_train_accuracy):
    if train_accuracy >= best_train_accuracy: 
        print(f'Model Saved for {train_accuracy}')
        torch.save(model.state_dict(), f'{SAVE_PATH}_weights_{time}.pth')

def evaluate_model(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.float().to(device).squeeze()

            outputs = model(images).squeeze()
            outputs = custom_function(outputs)
            
            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * (correct / total)

# Training loop
train_losses = []
train_accuracies = []
best_train_accuracy = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        # Forward pass
        images, labels = images.to(device), labels.float().to(device).squeeze()
       
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    train_accuracy = evaluate_model(train_loader)
    train_accuracies.append(train_accuracy)

    if train_accuracy > best_train_accuracy: best_train_accuracy = train_accuracy

    print(f'Epoch [{epoch}/{num_epochs}], Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Best Train Accuracy: {best_train_accuracy:.2f}%', flush=True)
    plot(train_losses, train_accuracies, train_accuracy, best_train_accuracy)
        

# Save the model
torch.save(model.state_dict(), f'{SAVE_PATH}_final_weights_{time}.pth')
