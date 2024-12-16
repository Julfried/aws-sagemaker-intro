import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    Simple CNN model for MNIST classification

    From keras tutorial: https://keras.io/examples/vision/mnist_convnet/
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc2 = nn.Linear(1600, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Convolutional layers
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        # Fully connected layers
        x = torch.flatten(x, 1)
        x = torch.dropout(x, 0.5, self.training)
        x = self.fc2(x)
        return self.softmax(x) if not self.training else x # Dont apply softmax during training because Loss functions in pytorch do not expect probabilities