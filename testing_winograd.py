from winograd2d import WinogradConv2d
from torchvision import datasets
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
from helper_functions import accuracy_fn
from train_test_loop import test_step, train_step

from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.auto import tqdm
from timeit import default_timer as timer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


batch_size = 32
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
classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")


print(train_loader)
print(train_data)

win_layer = WinogradConv2d(
    in_channels=1, out_channels=16, kernel_size=3, padding=1, stride=1
)

input_tensor = torch.randn(batch_size, 3, 28, 28)
output_tensor = win_layer.forward(input_tensor)


# class network(nn.Module):
#     def __init__(self):
#         super(network, self).__init__()

#         self.winograd1 = WinogradConv2d(in_channels=1, out_channels=16, m=3, r=1)

#     def forward(self, x):
#         # conv1
#         x = self.winograd1(x)

#         return x


# cnn = network().to(device)

# print(cnn)

# loss_function = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(cnn.parameters(), lr=0.01, momentum=0.9)


# train_time_start_on_device = timer()

# writer = SummaryWriter("runs/mnist_trainer_{}".format(train_time_start_on_device))

# epochs = 5
# for epoch in tqdm(range(epochs)):
#     print(f"Epoch: {epoch}\n-------- ")

#     """Training Step """
#     train_loss, train_acc = train_step(
#         model=cnn,
#         data_loader=train_loader,
#         loss_fn=loss_function,
#         optimizer=optimizer,
#         accuracy_fn=accuracy_fn,
#         device=device,
#     )

#     """Testing Step """
#     test_loss, test_acc = test_step(
#         model=cnn,
#         data_loader=test_loader,
#         loss_fn=loss_function,
#         accuracy_fn=accuracy_fn,
#         device=device,
#     )
#     print(f"epoch {epoch} done!")

#     writer.add_scalar("Loss/train", train_loss, epoch)
#     writer.add_scalar("Loss/test", test_loss, epoch)
#     writer.add_scalar("Accuracy/train", train_acc, epoch)
#     writer.add_scalar("Accuracy/test", test_acc, epoch)

# train_time_end_on_device = timer()

# runtime = train_time_end_on_device - train_time_start_on_device
# print(f"Total training runntime:{runtime} second on {device}")
