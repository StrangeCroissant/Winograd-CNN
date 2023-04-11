from accuracy_fn import accuracy_fn
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from tqdm.auto import tqdm
import numpy as np

# ignore noisy warnings
warnings.filterwarnings("ignore")
# device agnostic script runs in cuda if able
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_model(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    accuracy_fn,
    device=device,
):
    """
    Return a dict containing the results of model
    predicting data_loader.
    """
    # initiate metrics at 0
    loss, acc = 0, 0
    model = model.to(device)
    model.eval()

    with torch.inference_mode():
        for X, y in tqdm(data_loader):
            #
            X, y = X.to(device), y.to(device)

            # predict
            y_pred = model(X)

            # accumulate the loss and acc values per batch

            loss += loss_fn(y_pred, y)

            acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        # scale loss and acc to find the average loss per batch

        loss /= len(data_loader)
        acc /= len(data_loader)

    return {
        "model_name": model.__class__.__name__,
        "model_loss": loss.item(),
        "model_acc": acc,
    }
