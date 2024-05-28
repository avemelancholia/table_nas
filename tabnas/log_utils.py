import re
from collections import defaultdict
import numpy as np


def calculate_means(values):
    current_epoch = 0
    current_list = []
    output = []
    for epoch, metric in values:
        if epoch != current_epoch:
            output.append(np.mean(current_list))
            current_epoch = epoch
            current_list = [metric]
        else:
            current_list.append(metric)
    return output


def parse_log(file_name):
    metrics = {}

    with open(file_name, "r") as f:
        data = f.readlines()

    results = defaultdict(list)
    for line in data:
        epoch = re.search(r"Epoch [0-9]+", line)
        train_acc = re.search(r"Train accuracy: [0-9]*[.]?[0-9]+", line)
        val_acc = re.search(r"Validation accuracy: [0-9]*[.]?[0-9]+", line)
        train_loss = re.search(r"Train loss: [0-9]*[.]?[0-9]+", line)
        val_loss = re.search(r"validation loss: [0-9]*[.]?[0-9]+", line)
        test_acc = re.search(r"top-1 = [0-9]*[.]?[0-9]+", line)

        if epoch is not None:
            epoch = int(epoch.group(0).split(" ")[-1])

        if train_acc is not None:
            train_acc = float(train_acc.group(0).split(" ")[-1])

        if val_acc is not None:
            val_acc = float(val_acc.group(0).split(" ")[-1])

        if train_loss is not None:
            train_loss = float(train_loss.group(0).split(" ")[-1])

        if val_loss is not None:
            val_loss = float(val_loss.group(0).split(" ")[-1])

        if test_acc is not None:
            test_acc = float(test_acc.group(0).split(" ")[-1])
            results["test_acc"] = test_acc

        if epoch is not None and train_acc is not None:
            results["train_acc"].append((epoch, train_acc))
            results["val_acc"].append((epoch, val_acc))

        if epoch is not None and train_loss is not None:
            results["train_loss"].append((epoch, train_loss))
            results["val_loss"].append((epoch, val_loss))

    metrics["train_acc"] = calculate_means(results["train_acc"])
    metrics["val_acc"] = calculate_means(results["val_acc"])
    metrics["train_loss"] = calculate_means(results["train_loss"])
    metrics["val_loss"] = calculate_means(results["val_loss"])
    metrics["test_acc"] = results["test_acc"]

    return metrics
