import mnist
import torch
import torch.nn as nn
import torch.optim as optim
from winograd2d import WinogradConv2d
from torchvision import datasets
from torchvision.transforms import transforms
import warnings
from helper_functions import accuracy_fn
from train_test_loop import test_step, train_step
from tqdm.auto import tqdm
from timeit import default_timer as timer

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
winograd_convolution = WinogradConv2dF23(
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

"""
Initializing training-testing of nn 
"""
train_time_start_on_device = timer()


epochs = 3
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-------- ")

    """Training Step """
    train_step(
        model=model,
        data_loader=train_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn,
        device=device,
    )

    """Testing Step """
    test_step(
        model=model,
        data_loader=test_loader,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        device=device,
    )
    print(f"epoch {epoch} done!")


train_time_end_on_device = timer()

runtime = train_time_end_on_device - train_time_start_on_device
print(f"Total training runntime:{runtime} second")
