

import torch

def load_model(server_model, client_models, file_path='model.pth'):
    server_model.load_state_dict(torch.load('./server/server_' + file_path))
    for i, client in enumerate(client_models):
        client.load_state_dict(torch.load(f'./clients/client_{i+1}_' + file_path))
    
    server_model.eval()
    for client in client_models:
        client.eval()

def test(server_model, client_models, test_loader):
    server_model.eval()
    for client in client_models:
        client.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.view(data.size(0), -1)  
            server_output = server_model(data)
            client_outputs = []
            for client in client_models:
                client_output = client(server_output.detach())  
                client_outputs.append(client_output)
            final_output = torch.mean(torch.stack(client_outputs), dim=0)
            _, predicted = torch.max(final_output, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")


# Save the trained model (including server and client models)
def save_model(server_model, client_models, file_path='model.pth'):
    torch.save(server_model.state_dict(), 'server_' + file_path)
    for i, client in enumerate(client_models):
        torch.save(client.state_dict(), f'client_{i+1}_' + file_path)

