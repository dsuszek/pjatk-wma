from __future__ import annotations
import argparse
import logging
import os
import time
import torch
import matplotlib.pyplot as plt
from torch.nn import Module, Conv2d, MaxPool2d, Linear, ReLU, LogSoftmax
from torch.utils.data import DataLoader, random_split
from torch import flatten
from torch.optim import Adam
from torchvision import transforms
from torchvision.datasets import ImageFolder

# ==================================================================================================
#                                            LOGGER
# ==================================================================================================

logger = logging.getLogger()
lprint = logger.info


def setup_logger(dataset_path: str | None = None) -> None:
    log_formatter = logging.Formatter('%(message)s')

    if dataset_path:
        logfile_path = os.path.join(dataset_path, 'dataset.log')
        file_handler = logging.FileHandler(logfile_path)
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    logger.setLevel(logging.INFO)


def print_text_separator():
    lprint('--------------------------------------------------------')


# ==================================================================================================
#                                         NEURAL NETWORKS
# ==================================================================================================
class LeNet(Module):
    def __init__(self, input_shape: torch.Size, classes: int):
        # call the parent constructor
        super(LeNet, self).__init__()
        channel_count = input_shape[0]
        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = Conv2d(in_channels=channel_count, out_channels=20,
                            kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # [(W−K+2P)/S]+1
        conv_out = int((input_shape[1] - 5) / 2) + 1

        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = Conv2d(in_channels=20, out_channels=50,
                            kernel_size=(5, 5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        conv_out = int((conv_out - 5) / 2) + 1
        conv_size = conv_out * conv_out * 50
        lprint(conv_out)

        # initialize first (and only) set of FC => RELU layers
        self.fc1 = Linear(in_features=conv_size, out_features=500)
        self.relu3 = ReLU()
        # initialize our softmax classifier
        self.fc2 = Linear(in_features=500, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.fc2(x)
        output = self.logSoftmax(x)
        # return the output predictions
        return output


# ==================================================================================================
#                                         TRAINING WRAPPER 
# ==================================================================================================

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_path', type=str, required=True)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-t', '--train_split', type=float, default=0.7)
    parser.add_argument('--initial_learning_rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=3)
    return parser.parse_args()


def get_data_loaders(dataset_path: str, train_split: float, batch_size: int):
    transform = transforms.Compose([transforms.ToTensor()])
    img_folder = ImageFolder(dataset_path, transform=transform)
    lprint(f'Image folder length {len(img_folder)}')
    training_samples_count = int(len(img_folder) * train_split)
    validation_samples_count = int(len(img_folder) * (1.0 - train_split))
    training_samples_count += len(img_folder) - (training_samples_count + validation_samples_count)
    (train_data, val_data) = random_split(img_folder,
                                          [training_samples_count,
                                           validation_samples_count],
                                          generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_data, shuffle=True,
                              batch_size=batch_size)
    validation_loader = DataLoader(val_data, batch_size=batch_size)
    return train_loader, validation_loader, img_folder.classes


def train_network(initial_learning_rate: float, epochs: int,
                  train_loader: DataLoader, validation_loader: DataLoader,
                  classes: list[str]):
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    lprint(f'Device -> {device_name}')
    device = torch.device(device_name)
    batch_size = train_loader.batch_size
    train_steps = len(train_loader.dataset) // train_loader.batch_size
    validation_steps = len(validation_loader.dataset) // validation_loader.batch_size
    input_shape = next(iter(train_loader))[0][0].shape
    model = LeNet(input_shape, len(classes)).to(device)
    opt = Adam(model.parameters(), lr=initial_learning_rate)
    lossFn = torch.nn.NLLLoss()
    lprint('Initializing training')
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    print("[INFO] training the network...")
    startTime = time.time()
    # loop over our epochs
    for e in range(0, epochs):
        # set the model in training mode
        model.train()
        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0
        # initialize the number of correct predictions in the training
        # and validation step
        trainCorrect = 0
        valCorrect = 0
        # loop over the training set
        for (x, y) in train_loader:
            # send the input to the device
            (x, y) = (x.to(device), y.to(device))
            # perform a forward pass and calculate the training loss
            pred = model(x)
            loss = lossFn(pred, y)
            # zero out the gradients, perform the backpropagation step,
            # and update the weights
            opt.zero_grad()
            loss.backward()
            opt.step()
            # add the loss to the total training loss so far and
            # calculate the number of correct predictions
            totalTrainLoss += loss
            trainCorrect += (pred.argmax(1) == y).type(
                torch.float).sum().item()

        # switch off autograd for evaluation
        with torch.no_grad():
            # set the model in evaluation mode
            model.eval()
            # loop over the validation set
            for (x, y) in validation_loader:
                # send the input to the device
                (x, y) = (x.to(device), y.to(device))
                # make the predictions and calculate the validation loss
                pred = model(x)
                totalValLoss += lossFn(pred, y)
                # calculate the number of correct predictions
                valCorrect += (pred.argmax(1) == y).type(
                    torch.float).sum().item()

        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / train_steps
        avgValLoss = totalValLoss / (1.0 - train_steps)
        # calculate the training and validation accuracy
        trainCorrect = trainCorrect / len(train_loader.dataset)
        valCorrect = valCorrect / len(validation_loader.dataset)
        # update our training history
        history["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        history["train_acc"].append(trainCorrect)
        history["val_loss"].append(avgValLoss.cpu().detach().numpy())
        history["val_acc"].append(valCorrect)
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, epochs))
        print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
            avgTrainLoss, trainCorrect))
        print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
            avgValLoss, valCorrect))

    # finish measuring how long training took
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        endTime - startTime))
    return model, history


def visualisation_of_history(history):
    plt.title('Accuracy')
    plt.plot(history['train_acc'], '-', label='Train')
    plt.plot(history['val_acc'], '--', label='Validation')
    plt.legend()
    plt.show()


def main(args):
    setup_logger()
    train_loader, validation_loader, classes = get_data_loaders(args.dataset_path, args.train_split, args.batch_size)
    model, history = train_network(args.initial_learning_rate, args.epochs, train_loader, validation_loader, classes)
    visualisation_of_history(history)


if __name__ == '__main__':
    main(parse_arguments())
