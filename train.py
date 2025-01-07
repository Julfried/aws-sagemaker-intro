import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import pathlib
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import SimpleCNN

# Training loop
def train(model, loader, optimizer, loss_fn, device):
    model.train()
    for epoch in range(5):  # Number of epochs
        running_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(loader)}")

# Evaluation of the model
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct / total:.2f}%')


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR']) # SM_MODEL_DIR is an environment variable created by SageMaker that specifies the directory where the model artifacts are stored
    args = parser.parse_args()

    # Define data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Create DataLoader
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train and evaluate the model
    train(model, train_loader, optimizer, loss_fn, device)
    evaluate(model, test_loader, device)

    # Save the model to the correct output directory
    output_path = os.path.join(args.model_dir, 'model.pth')
    with open(output_path, 'wb') as f:
        torch.save(model.state_dict(), f)
    print(f"Model saved to {output_path}")


