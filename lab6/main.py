import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)


def train_model(model, optimizer, loss_function, train_dataloader, num_epochs=5):
    model.train()
    print(f"Training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        total_loss = 0
        for batch, (image_batch, labels_batch) in enumerate(train_dataloader):
            image_batch = image_batch.to(device)
            labels_batch = labels_batch.to(device)

            pred = model(image_batch)
            loss = loss_function(pred, labels_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if batch % 100 == 0:
                print(f"Epoch {epoch+1}, Batch index {batch}, Loss: {loss.item():>7f}")

        print(f"Epoch {epoch+1} finished. Average Loss: {total_loss / len(train_dataloader):>7f}")


def test_model(model, loss_function, test_dataloader):
    model.eval()
    correct = 0.
    test_loss = 0.
    size = len(test_dataloader.dataset)
    with torch.no_grad():
        for image_batch, labels_batch in test_dataloader:
            image_batch = image_batch.to(device)
            labels_batch = labels_batch.to(device)
            pred = model(image_batch)
            test_loss += loss_function(pred, labels_batch).item()
            correct += (pred.argmax(1) == labels_batch).type(torch.float).sum().item()

    test_loss /= len(test_dataloader)
    correct /= size
    print(f"Test Accuracy: {(100*correct):>0.1f}%, Test Loss: {test_loss:>8f} \n")


class NeuralNetwork(nn.Module):
    def __init__(self, hidden_layer_sizes, activation_function):
        super().__init__()
        self.flatten = nn.Flatten()
        layers = []
        layers.append(nn.Linear(28 * 28, hidden_layer_sizes[0]))
        layers.append(activation_function)

        for i in range(len(hidden_layer_sizes) - 1):
            layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i+1]))
            layers.append(activation_function)

        layers.append(nn.Linear(hidden_layer_sizes[-1], 10))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)


def main():
    loss_function = nn.CrossEntropyLoss()

    print("Training model A...")
    model_a = NeuralNetwork(hidden_layer_sizes=[1], activation_function=nn.Tanh()).to(device)
    optimizer_a = optim.SGD(model_a.parameters(), lr=1e-2)
    train_model(model_a, optimizer_a, loss_function, train_dataloader)
    test_model(model_a, loss_function, test_dataloader)

    print("Training model B...")
    model_b = NeuralNetwork(hidden_layer_sizes=[10], activation_function=nn.Tanh()).to(device)
    optimizer_b = optim.SGD(model_b.parameters(), lr=1e-2)
    train_model(model_b, optimizer_b, loss_function, train_dataloader)
    test_model(model_b, loss_function, test_dataloader)

    print("Training model C...")
    model_c = NeuralNetwork(hidden_layer_sizes=[10], activation_function=nn.Tanh()).to(device)
    optimizer_c = optim.SGD(model_c.parameters(), lr=1e-5)
    train_model(model_c, optimizer_c, loss_function, train_dataloader)
    test_model(model_c, loss_function, test_dataloader)

    print("Training model D...")
    model_d = NeuralNetwork(hidden_layer_sizes=[10], activation_function=nn.Tanh()).to(device)
    optimizer_d = optim.SGD(model_d.parameters(), lr=10)
    train_model(model_d, optimizer_d, loss_function, train_dataloader)
    test_model(model_d, loss_function, test_dataloader)

    print("Training model E...")
    model_e = NeuralNetwork(hidden_layer_sizes=[10, 10], activation_function=nn.Tanh()).to(device)
    optimizer_e = optim.SGD(model_e.parameters(), lr=1e-2)
    train_model(model_e, optimizer_e, loss_function, train_dataloader)
    test_model(model_e, loss_function, test_dataloader)

    print("Training model F...")
    model_f = NeuralNetwork(hidden_layer_sizes=[10, 10], activation_function=nn.ReLU()).to(device)
    optimizer_f = optim.SGD(model_f.parameters(), lr=1e-2)
    train_model(model_f, optimizer_f, loss_function, train_dataloader)
    test_model(model_f, loss_function, test_dataloader)

    print("Training model G...")
    model_g = NeuralNetwork(hidden_layer_sizes=[100, 100], activation_function=nn.ReLU()).to(device)
    optimizer_g = optim.SGD(model_g.parameters(), lr=1e-2)
    train_model(model_g, optimizer_g, loss_function, train_dataloader)
    test_model(model_g, loss_function, test_dataloader)

    print("Training model H...")
    model_h = NeuralNetwork(hidden_layer_sizes=[100, 100], activation_function=nn.ReLU()).to(device)
    optimizer_h = optim.SGD(model_h.parameters(), lr=1e-2, momentum=0.9)
    train_model(model_h, optimizer_h, loss_function, train_dataloader)
    test_model(model_h, loss_function, test_dataloader)

if __name__ == "__main__":
    main()
