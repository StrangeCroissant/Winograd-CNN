import mnist
import torch
import torch.nn as nn
import torch.optim as optim
from winograd2d import WinogradConv2d
from torchvision import datasets
from torchvision.transforms import transforms
import warnings

# ignore noisy warnings
warnings.filterwarnings("ignore")
# device agnostic script runs in cuda if able
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Set the input, output channel dimensions and batch size
batch_size = 32
in_channels = 16
out_channels = 32


train_data = datasets.MNIST(
    root="./data", train=True, download=True, transform=transforms.ToTensor()
)
test_data = datasets.MNIST(
    root="./data", train=False, download=True, transform=transforms.ToTensor()
)

# data loaders

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=False
)

print(train_loader)
print(train_data)

# intantiate winograd layer
winograd_convolution = WinogradConv2d(
    in_channels=in_channels, out_channels=out_channels
).to(device)


model = nn.Sequential(
    winograd_convolution,
    nn.BatchNorm2d(out_channels),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(out_channels, out_channels * 2, 3, padding=1),
    nn.BatchNorm2d(out_channels * 2),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(out_channels * 2 * 7 * 7, 10),
).to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


print(model)
# train loop
from accuracy_fn import accuracy_fn


def train_step(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    accuracy_fn,
    device: torch.device = device,
):
    # inititalize 0 acc and loss
    train_loss, train_acc = 0, 0

    # set model to train mode
    model.train()

    for batch, (X, y) in enumerate(data_loader):
        # move data to device
        X, y = X.to(device), y.to(device)

        # forward pass
        y_pred = model(X)
        # loss_per_batch
        loss = loss_fn(y_pred, y)

        # accumulate loss and accc
        train_loss += loss

        train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        # zero grad optim
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # train loss acc average per batch

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    print(f"Train loss:{train_loss:.5f} | Train acc:{train_acc:.2f}%")
