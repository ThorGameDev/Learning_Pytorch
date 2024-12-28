import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.models as models
import random

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 10),
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    itterer = int(0)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        itterer += 1
        if itterer >= 100:
            itterer = 0
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:7f} [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    test_loss, correct = 0,0
    
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, AVG loss: {test_loss:>8f} \n")
    return test_loss

model = NeuralNetwork()
#model.load_state_dict(torch.load("enhanced_optimization_model.pth"))

learning_rate = 1e-3 * 10
batch_size = 64
epochs = 20

loss_fn = nn.CrossEntropyLoss()

goal = test_loop(test_dataloader, model, loss_fn)

result = 999
for t in range(epochs):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate * (epochs - t) / epochs)
    print(f"Epoch {t+1}\n-------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    result = test_loop(test_dataloader, model, loss_fn)
print("done")

if goal > result:
    print("Goal was surpassed!")
    torch.save(model.state_dict(), "enhanced_optimization_model.pth")
