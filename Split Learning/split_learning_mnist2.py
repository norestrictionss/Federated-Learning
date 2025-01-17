import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Server Model (First part of the model with CNN)
class ServerModel(nn.Module):
    def __init__(self):
        super(ServerModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # Convolution layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Convolution layer
        self.pool = nn.MaxPool2d(2, 2)  # MaxPooling layer
        self.fc1 = nn.Linear(64 * 7 * 7, 256)  # Fully connected layer
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten the output of the convolutional layers
        x = self.fc1(x)
        return x

# Client Model (Second part of the model with Dropout and Linear layers)
class ClientModel(nn.Module):
    def __init__(self):
        super(ClientModel, self).__init__()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)  # Output size is 10 for classification (MNIST)
        self.dropout = nn.Dropout(p=0.5)  # Dropout layer for regularization

    def forward(self, x):
        x = self.fc2(x)
        x = self.dropout(x)  # Apply dropout during training
        x = self.fc3(x)
        return x

# Initialize the server model and multiple client models
server_model = ServerModel()
client_models = [ClientModel() for _ in range(3)]  # Example: 3 clients

# Initialize AdamW optimizer for server and clients
server_optimizer = optim.AdamW(server_model.parameters(), lr=0.001, weight_decay=1e-4)
client_optimizers = [optim.AdamW(client.parameters(), lr=0.001, weight_decay=1e-4) for client in client_models]

# Learning Rate Scheduler for dynamic adjustment
scheduler = optim.lr_scheduler.StepLR(server_optimizer, step_size=1, gamma=0.7)

# Loss function
criterion = nn.CrossEntropyLoss()

# Training loop for multiple clients
def train(server_model, client_models, server_optimizer, client_optimizers, scheduler, train_loader, num_clients=3):
    for epoch in range(1, 6):  # Train for 5 epochs
        epoch_loss = 0.0
        print(f"\n--- Epoch {epoch} ---")

        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.view(data.size(0), 1, 28, 28)  # Ensure data is in (N, C, H, W) format for CNN

            # Step 1: Server computes forward pass
            server_output = server_model(data)
            print(f"\nBatch {batch_idx + 1}: Server output (cut layer) summary - "
                  f"Mean: {server_output.mean().item():.4f}, Std: {server_output.std().item():.4f}, "
                  f"Sample: {server_output[0, :5].detach().numpy()}")

            client_losses = []

            # Step 2: Each client computes its forward pass and loss
            for i in range(num_clients):
                print(f"\nTraining Client {i + 1}...")

                # Client forward pass
                client_output = client_models[i](server_output.detach())  # Detach to avoid modifying server gradients
                loss = criterion(client_output, labels)

                print(f"Client {i + 1} output summary - "
                      f"Mean: {client_output.mean().item():.4f}, Std: {client_output.std().item():.4f}, "
                      f"Loss: {loss.item():.4f}")

                client_losses.append(loss)

                # Step 3: Backward pass for each client
                client_optimizers[i].zero_grad()
                loss.backward()
                client_optimizers[i].step()

            # Step 4: Aggregate client gradients and update server
            server_optimizer.zero_grad()
            total_loss = sum(loss.item() for loss in client_losses)
            epoch_loss += total_loss

            print(f"Batch {batch_idx + 1} Total Loss: {total_loss:.4f}")

        # Step 5: Update learning rate using the scheduler
        scheduler.step()

        print(f"Epoch {epoch} Average Loss: {epoch_loss / len(train_loader):.4f}")

# Example of using a DataLoader for training (e.g., MNIST dataset)
transform = transforms.Compose([transforms.RandomRotation(10),  # Data Augmentation: Random Rotation
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Train the model with multiple clients
train(server_model, client_models, server_optimizer, client_optimizers, scheduler, train_loader)
