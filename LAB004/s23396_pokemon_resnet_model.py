"""
This script defines a convolutional neural network (CNN)
model called PokemonCNN and trains it on a custom dataset using PyTorch library.
User can give the following input parameters:
    * dataset_path: path to the input dataset
    * batch size: defines the number of samples that will be propagated through the network
    * train_split: split between train and test datasets
    * epochs: number of complete iterations through the entire training dataset in one cycle

It also includes functions for setting up logging, parsing command-line arguments and loading data.
The implementation is based on ResNet model.

Example of usage:
    python pokemon_resnet_model.py --d path_to_dataset --b 32 --t 0.7 --i 0.001 --e 50
"""

from __future__ import annotations
import argparse
import logging
import os
import ssl
import csv
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn, optim
from torch.nn import Linear, ReLU, Conv2d, MaxPool2d, Module, LogSoftmax
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

logger = logging.getLogger()
lprint = logger.info
ssl._create_default_https_context = ssl._create_unverified_context


class PokemonCNN(Module):
    """
     CNN model based on ResNet architecture for image classification of Pokemons.
     """

    def __init__(self, input_shape: torch.Size, classes: int):
        """
        Initializes the PokemonCNN instance.

        Args:
            input_shape (torch.Size): The input shape of the data.
            classes (int): The number of classes for classification.
        """
        super(PokemonCNN, self).__init__()
        channel_count = input_shape[0]

        # Convolutional layers
        self.conv1 = Conv2d(in_channels=channel_count, out_channels=20, kernel_size=5)
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = Conv2d(in_channels=20, out_channels=50, kernel_size=5)
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        # Connect every input neuron to every output neuron
        conv_out_size = self._calculate_conv_out(input_shape[0])
        self.fc1 = Linear(conv_out_size, 500)
        self.relu3 = ReLU()
        self.fc2 = Linear(500, classes)
        self.logSoftmax = LogSoftmax(dim=1)

    def _calculate_conv_out(self, input_shape: int):
        """
        Calculates the output size after convolutional layers.

        Args:
            input_shape: The input shape to be used for calculating output size.

        Returns:
            torch.Size: The computed output size after convolutional layers.
        """
        output_shape = torch.randn(1, input_shape)
        output_shape = self.conv1(output_shape)
        output_shape = self.relu1(output_shape)
        output_shape = self.maxpool1(output_shape)
        output_shape = self.conv2(output_shape)
        output_shape = self.relu2(output_shape)
        output_shape = self.maxpool2(output_shape)
        return output_shape.view(output_shape.size(0), -1)

    def forward(self, input_tensor):
        """
        Defines the forward pass of the model.

        Args:
            input_tensor (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor from the forward pass.
        """
        input_tensor = self.cnn_layers(input_tensor)
        input_tensor = input_tensor.view(input_tensor.size(0), -1)
        input_tensor = self.linear_layers(input_tensor)
        return input_tensor


def initialize_resnet():
    """
    Initializes a ResNet model, that is built in torchvision library.

    Returns:
        torchvision.models.ResNet: The initialized ResNet model.
    """
    resnet_model = torchvision.models.resnet18()
    return resnet_model


def setup_logger(dataset_path: str | None = None) -> None:
    """
    Configures the logger.

    Args:
        dataset_path (str, optional): The path to the dataset directory. Defaults to None.
    """
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


def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_path', type=str, required=True)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-t', '--train_split', type=float, default=0.7)
    parser.add_argument('-i', '--initial_learning_rate', type=float, default=1e-3)
    parser.add_argument('-e', '--epochs', type=int, default=50)
    return parser.parse_args()


def get_data_loaders(dataset_path: str, train_split: float, batch_size: int):
    """
     Creates data loaders for the training and validation datasets.

     Args:
         dataset_path (str): The path to the dataset directory.
         train_split (float): Percentage of data used to train the model.
         batch_size (int): The batch size for data loaders.

     Returns:
         tuple: A tuple containing train_loader, validation_loader, and classes.
     """
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


def train_model(model, initial_learning_rate, epochs: int,
                train_loader: DataLoader, validation_loader: DataLoader):
    """
     Trains the specified model using the provided initial learning rate, epochs and data loaders.

     Args:
         model: The model to be trained.
         initial_learning_rate (float): The initial learning rate for optimization.
         epochs (int): The number of epochs for training.
         train_loader (DataLoader): The data loader for training data.
         validation_loader (DataLoader): The data loader for validation data.

     Returns:
         tuple: A tuple containing the trained model and the training history.
     """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    test_accuracy_max = -1

    for epoch in range(0, epochs):
        correct = 0
        total = 0
        total_train_loss = 0
        total_test_loss = 0

        for images, labels in train_loader:
            (images, labels) = (images.to(device), labels.to(device))

            optimizer.zero_grad()

            output = model.forward(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            output.squeeze_(-1)
            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)
        history["train_loss"].append(train_loss)
        train_accuracy = 100 * correct / total
        history["train_acc"].append(train_accuracy)
        correct_t = 0
        total_t = 0
        with torch.no_grad():
            for data in validation_loader:
                images_t, labels_t = data
                images_t = images_t.to(device)
                output_t = model.forward(images_t)
                _, predicted_t = torch.max(output_t.data, 1)
                total_t += labels_t.size(0)
                correct_t += (predicted_t == labels_t).sum().item()
                output_t.squeeze_(-1)
                loss_t = loss_function(output_t, labels_t)
                total_test_loss += loss_t.item()

        val_loss = total_test_loss / len(validation_loader)
        history["val_loss"].append(val_loss)
        val_accuracy = 100 * correct_t / total_t
        history["val_acc"].append(val_accuracy)
        print('Epoch %d:\nTrain loss: %.4f' % (epoch, loss.item()))
        print('Test loss: %.4f' % (loss_t.item()))
        print('Train_accuracy %.2f' % train_accuracy)
        print('Test_accuracy %.2f' % val_accuracy)
        if val_accuracy > test_accuracy_max:
            test_accuracy_max = val_accuracy
            print("New max test accuracy achieved %.2f. Saving model.\n\n" % test_accuracy_max)
            torch.save(model, 'best_cnn_pokemons.pth')
        else:
            print("Test accuracy did not increase from %.2f.\n\n" % test_accuracy_max)

    return model, history


def visualisation_of_history(history):
    """
    Plots the training and validation accuracy over epochs.

    Args:
        history (dict): A dictionary containing training and validation metrics.
            The dictionary should have the following keys:
                - 'train_acc': A list of training accuracy values for each epoch.
                - 'val_acc': A list of validation accuracy values for each epoch.

    This function creates a plot with the training accuracy and validation accuracy
    on the y-axis and the number of epochs on the x-axis. The plot includes a legend
    to differentiate between training and validation accuracy. The training accuracy
    is plotted as a solid line, and the validation accuracy is plotted as a dashed line.
    """
    plt.title('Accuracy')
    plt.plot(history['train_acc'], '-', label='Train')
    plt.plot(history['val_acc'], '--', label='Validation')
    plt.legend()
    plt.show()

    plt.title('Loss')
    plt.plot(history['train_loss'], '-', label='Train')
    plt.plot(history['val_loss'], '--', label='Validation')
    plt.legend()
    plt.show()


def main(args):
    """
    Starts all the functions relevant for training process.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    setup_logger()
    train_loader, validation_loader, classes = (
        get_data_loaders(args.dataset_path, args.train_split, args.batch_size))
    model = initialize_resnet()
    trained_model, history = train_model(model, args.initial_learning_rate,
                                         args.epochs, train_loader, validation_loader)

    with open('training_history.csv', 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for epoch in range(args.epochs):
            writer.writerow({
                'epoch': epoch + 1,
                'train_loss': history['train_loss'][epoch],
                'train_acc': history['train_acc'][epoch],
                'val_loss': history['val_loss'][epoch],
                'val_acc': history['val_acc'][epoch]
            })
    visualisation_of_history(history)


if __name__ == '__main__':
    main(parse_arguments())
