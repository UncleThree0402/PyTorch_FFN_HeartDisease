import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import sklearn.metrics as skm
from sklearn.metrics import classification_report

plt.rcParams["figure.figsize"] = (9, 6)


def box_plot(data, title):
    fig, ax = plt.subplots(1)
    ax = sns.boxplot(data=data)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_title(title)
    plt.show()


def data_2_data_dl(data=None, label=None, train_size=0.8, ratio_valid=0.5):
    # Split
    train_x, testval_x, train_y, testval_y = train_test_split(data, label, train_size=0.8)
    valid_x, test_x, valid_y, test_y = train_test_split(testval_x, testval_y, train_size=0.5)

    # Dataset
    train_set = TensorDataset(train_x, train_y)
    valid_set = TensorDataset(valid_x, valid_y)
    test_set = TensorDataset(test_x, test_y)

    # DataLoader
    train_dl = DataLoader(train_set, shuffle=True, batch_size=8, drop_last=True)
    valid_dl = DataLoader(valid_set, shuffle=False, batch_size=valid_set.tensors[1].shape[0])
    test_dl = DataLoader(test_set, shuffle=False, batch_size=test_set.tensors[1].shape[0])

    return train_dl, valid_dl, test_dl


def accuracy(y_pred, y_true):
    pred = nn.Sigmoid()(y_pred) > 0.5
    return torch.sum(pred == y_true) / len(y_pred)


def plot_loss(train_losses, valid_losses):
    plt.plot(range(len(train_losses)), train_losses)
    plt.plot(range(len(valid_losses)), valid_losses)
    plt.title("Losses")
    plt.xlabel("Epochs")
    plt.ylabel(":oss")
    plt.legend(["train", "valid"])
    plt.show()


def plot_accuracy(train_accuracies, valid_accuracies):
    plt.plot(range(len(train_accuracies)), train_accuracies)
    plt.plot(range(len(valid_accuracies)), valid_accuracies)
    plt.title("Accuracies")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["train", "valid"])
    plt.show()


def plot_lr(lr_rate):
    plt.plot(range(len(lr_rate)), lr_rate)
    plt.title("Learning Rate")
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.show()


def show_performance_binary(train_dl, valid_dl, test_dl, model):
    train_metrics = [0, 0, 0, 0]
    valid_metrics = [0, 0, 0, 0]
    test_metrics = [0, 0, 0, 0]

    model.eval()
    with torch.no_grad():
        y_pred_train = model(train_dl.dataset.tensors[0])
        y_pred_train = nn.Sigmoid()(y_pred_train)

        y_pred_valid = model(valid_dl.dataset.tensors[0])
        y_pred_valid = nn.Sigmoid()(y_pred_valid)

        y_pred_test = model(test_dl.dataset.tensors[0])
        y_pred_test = nn.Sigmoid()(y_pred_test)

    train_metrics[0] = skm.accuracy_score(train_dl.dataset.tensors[1], y_pred_train > 0.5)
    train_metrics[1] = skm.precision_score(train_dl.dataset.tensors[1], y_pred_train > 0.5)
    train_metrics[2] = skm.recall_score(train_dl.dataset.tensors[1], y_pred_train > 0.5)
    train_metrics[3] = skm.f1_score(train_dl.dataset.tensors[1], y_pred_train > 0.5)

    valid_metrics[0] = skm.accuracy_score(valid_dl.dataset.tensors[1], y_pred_valid > 0.5)
    valid_metrics[1] = skm.precision_score(valid_dl.dataset.tensors[1], y_pred_valid > 0.5)
    valid_metrics[2] = skm.recall_score(valid_dl.dataset.tensors[1], y_pred_valid > 0.5)
    valid_metrics[3] = skm.f1_score(valid_dl.dataset.tensors[1], y_pred_valid > 0.5)

    test_metrics[0] = skm.accuracy_score(test_dl.dataset.tensors[1], y_pred_test > 0.5)
    test_metrics[1] = skm.precision_score(test_dl.dataset.tensors[1], y_pred_test > 0.5)
    test_metrics[2] = skm.recall_score(test_dl.dataset.tensors[1], y_pred_test > 0.5)
    test_metrics[3] = skm.f1_score(test_dl.dataset.tensors[1], y_pred_test > 0.5)

    plt.bar(np.arange(0, 4) - .1, train_metrics, width=0.5)
    plt.bar(np.arange(0, 4), valid_metrics, width=0.5)
    plt.bar(np.arange(0, 4) + .1, test_metrics, width=0.5)
    plt.xticks(np.arange(0, 4), ["Accuracy", "Precision", "Recall", "F1"])
    plt.legend(["Train", "Valid", "Test"])
    plt.ylim(0.5, 1)
    plt.title("Performance")
    plt.show()

    train_report = classification_report(train_dl.dataset.tensors[1], y_pred_train > 0.5, target_names=["No", "Yes"])
    valid_report = classification_report(valid_dl.dataset.tensors[1], y_pred_valid > 0.5, target_names=["No", "Yes"])
    test_report = classification_report(test_dl.dataset.tensors[1], y_pred_test > 0.5, target_names=["No", "Yes"])

    print(f"Training Data : \n {train_report}")
    print(f"Validating Data : \n {valid_report}")
    print(f"Testing Data : \n {test_report}")

    train_confm = skm.confusion_matrix(train_dl.dataset.tensors[1], y_pred_train > 0.5)
    valid_confm = skm.confusion_matrix(valid_dl.dataset.tensors[1], y_pred_valid > 0.5)
    test_confm = skm.confusion_matrix(test_dl.dataset.tensors[1], y_pred_test > 0.5)

    ax = plt.axes()
    ax.imshow(train_confm, "Blues", vmax=(len(y_pred_train) / 2))
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No", "Yes"])
    ax.set_yticklabels(["No", "Yes"])
    ax.set_xlabel("Prediction")
    ax.set_ylabel("True")
    ax.text(0, 0, train_confm[0, 0])
    ax.text(0, 1, train_confm[0, 1])
    ax.text(1, 0, train_confm[1, 0])
    ax.text(1, 1, train_confm[1, 1])
    ax.set_title("Train Confusion Matrix")
    plt.show()

    ax = plt.axes()
    ax.imshow(valid_confm, "Blues", vmax=(len(y_pred_valid) / 2))
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No", "Yes"])
    ax.set_yticklabels(["No", "Yes"])
    ax.set_xlabel("Prediction")
    ax.set_ylabel("True")
    ax.text(0, 0, valid_confm[0, 0])
    ax.text(0, 1, valid_confm[0, 1])
    ax.text(1, 0, valid_confm[1, 0])
    ax.text(1, 1, valid_confm[1, 1])
    ax.set_title("Valid Confusion Matrix")
    plt.show()

    ax = plt.axes()
    ax.imshow(test_confm, "Blues", vmax=(len(y_pred_test) / 2))
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No", "Yes"])
    ax.set_yticklabels(["No", "Yes"])
    ax.set_xlabel("Prediction")
    ax.set_ylabel("True")
    ax.text(0, 0, test_confm[0, 0])
    ax.text(0, 1, test_confm[0, 1])
    ax.text(1, 0, test_confm[1, 0])
    ax.text(1, 1, test_confm[1, 1])
    ax.set_title("Test Confusion Matrix")
    plt.show()
