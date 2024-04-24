import numpy as np

def normalize_column(col):
    return (col - col.mean()) / col.std()


def calc_bce_positive_weights(label_occurences, power, factor):
    adjusted_occurences = np.power((label_occurences / label_occurences.min()), power) * label_occurences.min()
    bce_positive_weights = adjusted_occurences.min() / adjusted_occurences
    weight_factor = factor / label_occurences.max()
    bce_positive_weights = bce_positive_weights / bce_positive_weights.min() * weight_factor

    return bce_positive_weights


import torch
import torch.nn as nn
from hypll.tensors import TangentTensor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, average_precision_score, mean_absolute_error

def h_evaluate_loss(model, val_loader, criterion, manifold, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            tangents = TangentTensor(data=inputs, man_dim=-1, manifold=manifold)
            manifold_inputs = manifold.expmap(tangents)

            outputs = model(manifold_inputs)

            loss = criterion(outputs.tensor, targets)
            running_loss += loss.item() * inputs.size(0)
    return running_loss / len(val_loader.dataset)


def h_evaluate_metrics(model, dataloader, manifold, device, threshold=0.5):
    '''accuracy_score, hamming_loss, precision_score, recall_score, f1_score, average_precision_score'''
    model.eval()
    all_predictions = []
    bin_predictions = []
    all_targets = []
    sigmoid = nn.Sigmoid()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            tangents = TangentTensor(data=inputs, man_dim=-1, manifold=manifold)
            manifold_inputs = manifold.expmap(tangents)

            outputs = sigmoid(model(manifold_inputs).tensor)

            all_predictions.append(outputs.cpu().numpy())
            bin_predictions.append((outputs > threshold).float().cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    bin_predictions = np.concatenate(bin_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    accuracy = accuracy_score(all_targets, bin_predictions)
    hamm = hamming_loss(all_targets, bin_predictions)
    precision = precision_score(all_targets, bin_predictions, average='micro', zero_division=0)
    sensitivity = recall_score(all_targets, bin_predictions, average='micro')
    f1 = f1_score(all_targets, bin_predictions, average='micro')
    aps = np.array([average_precision_score(gt, p) for (gt, p) in zip(all_targets.T, all_predictions.T)])

    return accuracy, hamm, precision, sensitivity, f1, aps


def h_evaluate_mae(model, dataloader, manifold, device):
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            tangents = TangentTensor(data=inputs, man_dim=-1, manifold=manifold)
            manifold_inputs = manifold.expmap(tangents)

            outputs = model(manifold_inputs).tensor

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    mae = mean_absolute_error(all_targets, all_predictions)

    return mae


def h_mini_evaluate_loss(model, val_loader, criterion, manifold, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            tangents = TangentTensor(data=inputs, man_dim=-1, manifold=manifold)
            manifold_inputs = manifold.expmap(tangents)

            outputs = model(manifold_inputs)

            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
    return running_loss / len(val_loader.dataset)


def h_mini_evaluate_metrics(model, dataloader, manifold, device):
    '''accuracy_score, hamming_loss, average_precision_score, precision_score, recall_score, f1_score'''
    model.eval()
    all_predictions = []
    bin_predictions = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            tangents = TangentTensor(data=inputs, man_dim=-1, manifold=manifold)
            manifold_inputs = manifold.expmap(tangents)

            outputs = model(manifold_inputs)

            max_indices = torch.argmax(outputs, dim=-1)
            one_hot = torch.zeros_like(outputs)
            one_hot.scatter_(-1, max_indices.unsqueeze(-1), 1)

            all_predictions.append(outputs.cpu().numpy())
            bin_predictions.append(one_hot.float().cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    bin_predictions = np.concatenate(bin_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    accuracy = accuracy_score(all_targets, bin_predictions)
    hamm = hamming_loss(all_targets, bin_predictions)
    precision = precision_score(all_targets, bin_predictions, average='micro', zero_division=0)
    sensitivity = recall_score(all_targets, bin_predictions, average='micro')
    f1 = f1_score(all_targets, bin_predictions, average='micro')
    aps = np.array([average_precision_score(gt, p) for (gt, p) in zip(all_targets.T, all_predictions.T)])

    return accuracy, hamm, precision, sensitivity, f1, aps



def evaluate_loss(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
    return running_loss / len(val_loader.dataset)


def evaluate_metrics(model, dataloader, device, threshold=0.5):
    '''accuracy_score, hamming_loss, average_precision_score, precision_score, recall_score, f1_score'''
    model.eval()
    all_predictions = []
    bin_predictions = []
    all_targets = []
    sigmoid = nn.Sigmoid()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = sigmoid(model(inputs))

            all_predictions.append(outputs.cpu().numpy())
            bin_predictions.append((outputs > threshold).float().cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    bin_predictions = np.concatenate(bin_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    accuracy = accuracy_score(all_targets, bin_predictions)
    hamm = hamming_loss(all_targets, bin_predictions)
    precision = precision_score(all_targets, bin_predictions, average='micro', zero_division=0)
    sensitivity = recall_score(all_targets, bin_predictions, average='micro')
    f1 = f1_score(all_targets, bin_predictions, average='micro')
    aps = np.array([average_precision_score(gt, p) for (gt, p) in zip(all_targets.T, all_predictions.T)])

    return accuracy, hamm, precision, sensitivity, f1, aps


def evaluate_mae(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    mae = mean_absolute_error(all_targets, all_predictions)

    return mae


def mini_evaluate_loss(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
    return running_loss / len(val_loader.dataset)


def mini_evaluate_metrics(model, dataloader, device):
    '''accuracy_score, hamming_loss, average_precision_score, precision_score, recall_score, f1_score'''
    model.eval()
    all_predictions = []
    bin_predictions = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            max_indices = torch.argmax(outputs, dim=-1)
            one_hot = torch.zeros_like(outputs)
            one_hot.scatter_(-1, max_indices.unsqueeze(-1), 1)

            all_predictions.append(outputs.cpu().numpy())
            bin_predictions.append(one_hot.float().cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    bin_predictions = np.concatenate(bin_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    accuracy = accuracy_score(all_targets, bin_predictions)
    hamm = hamming_loss(all_targets, bin_predictions)
    precision = precision_score(all_targets, bin_predictions, average='micro', zero_division=0)
    sensitivity = recall_score(all_targets, bin_predictions, average='micro')
    f1 = f1_score(all_targets, bin_predictions, average='micro')
    aps = np.array([average_precision_score(gt, p) for (gt, p) in zip(all_targets.T, all_predictions.T)])

    return accuracy, hamm, precision, sensitivity, f1, aps
