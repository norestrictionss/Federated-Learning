import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from test_split import load_model, test, save_model

# Server Model (First part of the model)
class ServerModel(nn.Module):
    def __init__(self):
        super(ServerModel, self).__init__()
        self.layer1 = nn.Linear(784, 256)  # Input size is 784 (28x28 image flattened)
        self.layer2 = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

# Client Model (Second part of the model)
class ClientModel(nn.Module):
    def __init__(self):
        super(ClientModel, self).__init__()
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.ReLU()
        self.layer5 = nn.Linear(128, 10)  # Output size is 10 for classification (MNIST)

    def forward(self, x):
        x = self.layer4(self.layer3(x))
        x = self.layer5(x)
        return x

# Initialize the server model and multiple client models
server_model = ServerModel()
client_models = [ClientModel() for _ in range(3)]  # Example: 3 clients

# Initialize optimizers for the server and clients (using Adam instead of SGD)
server_optimizer = optim.Adam(server_model.parameters(), lr=0.001)
client_optimizers = [optim.Adam(client.parameters(), lr=0.001) for client in client_models]

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
            data = data.view(data.size(0), -1)  # Flatten the input images (e.g., 28x28 -> 784)

            server_output = server_model(data)
            print(f"\nBatch {batch_idx + 1}: Server output (cut layer) summary - "
                  f"Mean: {server_output.mean().item():.4f}, Std: {server_output.std().item():.4f}, "
                  f"Sample: {server_output[0, :5].detach().numpy()}")

            client_losses = []

            for i in range(num_clients):
                print(f"\nTraining Client {i + 1}...")

                # Client forward pass
                client_output = client_models[i](server_output.detach())  
                loss = criterion(client_output, labels)

                print(f"Client {i + 1} output summary - "
                      f"Mean: {client_output.mean().item():.4f}, Std: {client_output.std().item():.4f}, "
                      f"Loss: {loss.item():.4f}")

                client_losses.append(loss)
                client_optimizers[i].zero_grad()
                loss.backward()
                client_optimizers[i].step()
            server_optimizer.zero_grad()
            total_loss = sum(loss.item() for loss in client_losses)
            epoch_loss += total_loss

            print(f"Batch {batch_idx + 1} Total Loss: {total_loss:.4f}")
        scheduler.step()

        print(f"Epoch {epoch} Average Loss: {epoch_loss / len(train_loader):.4f}")


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)



# train(server_model, client_models, server_optimizer, client_optimizers, scheduler, train_loader)

# Test the model after training
# test(server_model, client_models, test_loader)

# Save the trained model to hardware
# save_model(server_model, client_models, 'trained_model.pth')


load_model(server_model, client_models, 'trained_model.pth')
test(server_model, client_models, test_loader)

# Loading the trained model from hardware


# Testing the model after loading it
# test(server_model, client_models, test_loader)