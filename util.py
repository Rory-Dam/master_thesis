import numpy as np

def normalize_column(col):
    return (col - col.mean()) / col.std()


def calc_bce_positive_weights(label_occurences, power, factor):
    adjusted_occurences = np.power((label_occurences / label_occurences.min()), power) * label_occurences.min()
    bce_positive_weights = adjusted_occurences.min() / adjusted_occurences
    weight_factor = factor / label_occurences.max()
    bce_positive_weights = bce_positive_weights / bce_positive_weights.min() * weight_factor

    return bce_positive_weights


def get_fold_indices(size, k):
    fold_size = size // k
    rest = size % k

    fold_sizes = [fold_size] * k

    for i in range(rest):
        fold_sizes[i] += 1

    indices = np.cumsum([fold_sizes])

    return list(zip(indices-np.array(fold_sizes), indices))


def get_fold_indices_rand(num_types, num_per_type, k, seed=42):
    def get_val_start_ends(size, k):
        fold_size = size // k
        rest = size % k

        fold_sizes = [fold_size] * k

        for i in range(rest):
            fold_sizes[i] += 1

        indices = np.cumsum([fold_sizes])

        return list(zip(indices-np.array(fold_sizes), indices))


    np.random.seed(seed)
    indices = np.random.random(num_types).argsort()

    val_start_ends = get_val_start_ends(num_types, k)
    val_indices = [indices[start:end] for start, end in val_start_ends]

    train_indices = [list(set(range(num_types)) - set(val_is)) for val_is in val_indices]
    exp_train_indices = [[list(range(val_i*num_per_type,(val_i+1)*num_per_type)) for val_i in val_is] for val_is in train_indices]

    return val_indices, [np.array(exp_is).flatten() for exp_is in exp_train_indices]


def get_subset_fold_indices_rand(frac, num_types, num_per_type, k, seed=42):
    def get_val_start_ends(size, k):
        fold_size = size // k
        rest = size % k

        fold_sizes = [fold_size] * k

        for i in range(rest):
            fold_sizes[i] += 1

        indices = np.cumsum([fold_sizes])

        return list(zip(indices-np.array(fold_sizes), indices))


    np.random.seed(seed)
    indices = np.random.random(num_types).argsort()
    indices = indices[:int(len(indices)*frac)]
    left_over_indices = np.setdiff1d(list(range(num_types)), indices)

    val_start_ends = get_val_start_ends(int(num_types*frac), k)
    val_indices = [np.append(indices[start:end], left_over_indices) for start, end in val_start_ends]

    train_indices = [list(set(range(num_types)) - set(val_is)) for val_is in val_indices]
    exp_train_indices = [[list(range(val_i*num_per_type,(val_i+1)*num_per_type)) for val_i in val_is] for val_is in train_indices]

    return val_indices, [np.array(exp_is).flatten() for exp_is in exp_train_indices]



import torch
import torch.nn as nn
from hypll.tensors import TangentTensor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, average_precision_score, mean_absolute_error, r2_score

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


def h_evaluate_mae_double_loss(model, dataloader, manifold, device):
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            targets, _ = targets[:,[0]], targets[:,[1]]

            tangents = TangentTensor(data=inputs, man_dim=-1, manifold=manifold)
            manifold_inputs = manifold.expmap(tangents)

            outputs = model(manifold_inputs).tensor

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    mae = mean_absolute_error(all_targets, all_predictions)

    return mae


def h_evaluate_r2(model, dataloader, manifold, device):
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

    r2 = r2_score(all_targets, all_predictions)

    return r2


def h_evaluate_r2_double_loss(model, dataloader, manifold, device):
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            targets, _ = targets[:,[0]], targets[:,[1]]

            tangents = TangentTensor(data=inputs, man_dim=-1, manifold=manifold)
            manifold_inputs = manifold.expmap(tangents)

            outputs = model(manifold_inputs).tensor

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    r2 = r2_score(all_targets, all_predictions)

    return r2


def h_multi_evaluate_r2(models, dataloader, manifold, device):
    all_predictions = []
    all_targets = []
    for model in models:
        model.eval()
        model_predictions = []
        model_targets = []
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)

                tangents = TangentTensor(data=inputs, man_dim=-1, manifold=manifold)
                manifold_inputs = manifold.expmap(tangents)

                outputs = model(manifold_inputs).tensor

                model_predictions.append(outputs.cpu().numpy())
                model_targets.append(targets.cpu().numpy())

        model_predictions = np.concatenate(model_predictions, axis=0)
        model_targets = np.concatenate(model_targets, axis=0)

        all_predictions.append(model_predictions)
        all_targets.append(model_targets)

    all_predictions = np.mean(all_predictions, axis=0)
    all_targets = np.mean(all_targets, axis=0)

    r2 = r2_score(all_targets, all_predictions)

    return r2


def h_evaluate_mae_classes(model, dataloader, manifold, device):
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

    maes = []
    for i in range(all_targets.shape[1]):
        maes.append(mean_absolute_error(all_targets[:,i], all_predictions[:,i]))

    return np.array(maes)


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


def evaluate_r2(model, dataloader, device):
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

    r2 = r2_score(all_targets, all_predictions)

    return r2


def multi_evaluate_r2(models, dataloader, device):
    all_predictions = []
    all_targets = []
    for model in models:
        model.eval()
        model_predictions = []
        model_targets = []
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)

                model_predictions.append(outputs.cpu().numpy())
                model_targets.append(targets.cpu().numpy())

        model_predictions = np.concatenate(model_predictions, axis=0)
        model_targets = np.concatenate(model_targets, axis=0)

        all_predictions.append(model_predictions)
        all_targets.append(model_targets)

    all_predictions = np.mean(all_predictions, axis=0)
    all_targets = np.mean(all_targets, axis=0)

    r2 = r2_score(all_targets, all_predictions)

    return r2


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
