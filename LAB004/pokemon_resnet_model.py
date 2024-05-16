from __future__ import annotations

import torch
import torchvision
import argparse
import logging
import os
from torch import nn, optim
from torch.nn import Linear, ReLU, Conv2d, MaxPool2d, Module, LogSoftmax
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
import ssl

logger = logging.getLogger()
lprint = logger.info
ssl._create_default_https_context = ssl._create_unverified_context


class PokemonCNN(Module):
    def __init__(self, input_shape: torch.Size, classes: int):
        # Call the parent constructor
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
        conv_out_size = self._calculate_conv_out(input_shape[0])
        self.fc1 = Linear(conv_out_size, 500)
        self.relu3 = ReLU()
        self.fc2 = Linear(500, classes)
        self.logSoftmax = LogSoftmax(dim=1)

    def _calculate_conv_out(self, input_shape):
        # Compute the output size after convolutional layers
        x = torch.randn(1, input_shape)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        return x.view(x.size(0), -1)

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


def initialize_resnet(num_classes):
    resnet_model = torchvision.models.resnet18()

    # Replace the final fully connected layer with a new one suitable for our number of classes
    num_ftrs = resnet_model.fc.in_features
    # resnet_model.fc = nn.Linear(num_ftrs, num_classes)

    return resnet_model


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


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_path', type=str, required=True)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-t', '--train_split', type=float, default=0.7)
    parser.add_argument('-i', '--initial_learning_rate', type=float, default=1e-3)
    parser.add_argument('-e', '--epochs', type=int, default=50)
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


def train_model(model, initial_learning_rate, epochs: int,
                train_loader: DataLoader, validation_loader: DataLoader):
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

            # Training pass
            optimizer.zero_grad()

            output = model.forward(images)
            _, predicted = torch.max(output.data, 1)  # the dimension 1 corresponds to max along the rows
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

        test_loss = total_test_loss / len(validation_loader.dataset)
        history["val_loss"].append(test_loss)
        test_accuracy = 100 * correct_t / total_t
        history["val_acc"].append(test_accuracy)
        print('Epoch %d:\ntrain loss: %.4f' % (epoch, loss.item()))
        print('test loss: %.4f' % (loss_t.item()))
        print('train_accuracy %.2f' % (train_accuracy))
        print('test_accuracy %.2f' % (test_accuracy))
        if test_accuracy > test_accuracy_max:
            test_accuracy_max = test_accuracy
            print("New Max Test Accuracy Achieved %.2f. Saving model.\n\n" % (test_accuracy_max))
            torch.save(model, 'best_test_acc_model.pth')
        else:
            print("Test accuracy did not increase from %.2f\n\n" % (test_accuracy_max))

    return model, history


def main(args):
    setup_logger()
    train_loader, validation_loader, classes = get_data_loaders(args.dataset_path, args.train_split, args.batch_size)
    model = initialize_resnet(classes)
    trained_model, history = train_model(model, args.initial_learning_rate, args.epochs, train_loader,
                                         validation_loader)


if __name__ == '__main__':
    main(parse_arguments())
