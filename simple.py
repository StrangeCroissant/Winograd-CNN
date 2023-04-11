import mnist
import torch
import torch.nn as nn
import torch.optim as optim
from winograd2d import WinogradConv2d
from torchvision import datasets
from torchvision.transforms import transforms
import torch.nn.functional as F

from accuracy_fn import accuracy_fn
from train_test_loop import test_step, train_step
from tqdm.auto import tqdm
from timeit import default_timer as timer

batch_size = 32
in_channels = 16
out_channels = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


class nn(torch.nn.Module):
    def __init__(self):
        super(nn, self).__init__()

        self.conv1 = WinogradConv2d(1, 10, m=2, r=1)
        self.fc1 = torch.nn.Linear(10 * 12 * 12, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = x.view(-1, 10 * 12 * 12)
        x = self.fc1(x)
        return x


model = nn()


train_time_start_on_device = timer()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


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
