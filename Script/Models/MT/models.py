import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GINConv, BatchNorm, GATv2Conv
from torch_geometric.nn.conv import GINEConv
from torch_geometric.nn.models.attentive_fp import GATEConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from sklearn.metrics import (f1_score, confusion_matrix, precision_score,
                              average_precision_score, fbeta_score, matthews_corrcoef)
from torch.optim import Adam

### BATCH NORMALIZATION SOLO SE USO CDK E XTB DESCRITTORI NUMERICI
## Qui la batch normalization viene rimossa

task_freq = {}


def set_class_frequencies(frequencies):
    global task_freq
    task_freq = frequencies
    for task, (n0, n1) in frequencies.items():
        print(f"  {task}: n0={n0}, n1={n1}")


def get_activation(name: str):
    return {
        "relu":       nn.ReLU(),
        "elu":        nn.ELU(),
        "leaky_relu": nn.LeakyReLU(0.1),
    }[name]


# =========================
# MODELS (invariati: versione SENZA cdk_dim)
# =========================

class GINModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout, out_channels=1, activation="relu"):
        super(GINModel, self).__init__()
        self.convs = nn.ModuleList()
        self.dropout = dropout
        self.act = get_activation(activation)

        mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.convs.append(GINConv(mlp))

        for _ in range(num_layers - 1):
            mlp_i = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            self.convs.append(GINConv(mlp_i))
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lin(x)
        return torch.sigmoid(x)


class GCNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout, out_channels=1, activation="relu"):
        super(GCNModel, self).__init__()
        self.convs = nn.ModuleList()
        self.dropout = dropout
        self.act = get_activation(activation)

        self.convs.append(GCNConv(in_channels, hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lin(x)
        return torch.sigmoid(x)


class GATC(nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_channels, num_layers, dropout, out_channels=1, heads=4, activation="relu"):
        super(GATC, self).__init__()
        self.dropout = dropout
        self.heads = heads
        self.out_heads = 1
        self.convs = nn.ModuleList()
        self.act = get_activation(activation)

        self.convs.append(GATConv(in_channels, hidden_channels, heads=self.heads, edge_dim=edge_dim))

        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * self.heads, hidden_channels, heads=self.heads, edge_dim=edge_dim))

        self.convs.append(GATConv(hidden_channels * self.heads, hidden_channels, heads=self.out_heads, edge_dim=edge_dim))
        self.lin = nn.Linear(hidden_channels * self.out_heads, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lin(x)
        return torch.sigmoid(x)


class GINEModel(nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_channels, num_layers,
                 dropout, out_channels=1, activation="relu"):
        super(GINEModel, self).__init__()
        self.convs = nn.ModuleList()
        self.dropout = dropout
        self.act = get_activation(activation)

        mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.convs.append(GINEConv(mlp, edge_dim=edge_dim))

        for _ in range(num_layers - 1):
            mlp_i = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            self.convs.append(GINEConv(mlp_i, edge_dim=edge_dim))

        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        return torch.sigmoid(x)


class AttentiveFPNode(nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_channels, num_layers,
                 dropout, out_channels=1, activation="relu"):
        super(AttentiveFPNode, self).__init__()
        self.dropout = dropout
        self.act = get_activation(activation)

        self.lin1 = nn.Linear(in_channels, hidden_channels)

        self.gate_conv = GATEConv(hidden_channels, hidden_channels,
                                  edge_dim, dropout)
        self.gru = nn.GRUCell(hidden_channels, hidden_channels)

        self.atom_convs = nn.ModuleList()
        self.atom_grus  = nn.ModuleList()
        for _ in range(num_layers - 1):
            conv = GATConv(hidden_channels, hidden_channels,
                           dropout=dropout, add_self_loops=False,
                           negative_slope=0.01)
            self.atom_convs.append(conv)
            self.atom_grus.append(nn.GRUCell(hidden_channels, hidden_channels))

        self.lin2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = F.leaky_relu(self.lin1(x))

        h = F.elu(self.gate_conv(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.gru(h, x).relu()

        for conv, gru in zip(self.atom_convs, self.atom_grus):
            h = conv(x, edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x).relu()

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return torch.sigmoid(x)


class GATv2Model(nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_channels, num_layers,
                 dropout, out_channels=1, heads=4, activation="relu"):
        super(GATv2Model, self).__init__()
        self.dropout = dropout
        self.heads = heads
        self.out_heads = 1
        self.convs = nn.ModuleList()
        self.act = get_activation(activation)

        self.convs.append(GATv2Conv(in_channels, hidden_channels,
                                    heads=self.heads, edge_dim=edge_dim,
                                    concat=True))
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_channels * self.heads, hidden_channels,
                                        heads=self.heads, edge_dim=edge_dim,
                                        concat=True))
        self.convs.append(GATv2Conv(hidden_channels * self.heads, hidden_channels,
                                    heads=self.out_heads, edge_dim=edge_dim,
                                    concat=False))

        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        return torch.sigmoid(x)


# =========================
# LOSS (invariata)
# =========================

def weighted_binary_crossentropy(y_true, y_pred, mask, task_names):
    bce_loss = F.binary_cross_entropy(y_pred, y_true, reduction='none')
    weights = torch.zeros_like(y_true)
    for t, task in enumerate(task_names):
        n0, n1 = task_freq[task]
        w0 = (n0 + n1) / (2 * n0) if n0 > 0 else 1.0
        w1 = (n0 + n1) / (2 * n1) if n1 > 0 else 1.0
        weights[:, t] = y_true[:, t] * w1 + (1 - y_true[:, t]) * w0
    masked_loss = bce_loss * weights * mask
    return masked_loss.sum() / mask.sum().clamp(min=1)


# =========================
# METRIC COMPUTATION (correct: on full set, not batch-averaged)
# =========================

def _compute_task_metrics(y_true, y_pred, y_prob):
    """
    Calcola tutte le metriche per un task SULL'INTERO SET (non mediate per batch).
    """
    empty = {"sensitivity": 0.0, "specificity": 0.0, "balanced_acc": 0.0,
             "precision": 0.0, "fbeta": 0.0, "auprc": float('nan'),
             "mcc": float('nan'), "acc": 0.0}
    if len(y_true) == 0:
        return empty

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    only_one_class = len(np.unique(y_true)) < 2

    if only_one_class:
        if y_true[0] == 0:
            tn = int((y_pred == 0).sum()); fp = int((y_pred == 1).sum())
            fn = 0; tp = 0
        else:
            tp = int((y_pred == 1).sum()); fn = int((y_pred == 0).sum())
            tn = 0; fp = 0
    else:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "sensitivity":  sens,
        "specificity":  spec,
        "balanced_acc": (sens + spec) / 2,
        "precision":    precision_score(y_true, y_pred, zero_division=0),
        "fbeta":        fbeta_score(y_true, y_pred, beta=0.5, zero_division=0),
        "auprc":        average_precision_score(y_true, y_prob) if not only_one_class else float('nan'),
        "mcc":          matthews_corrcoef(y_true, y_pred) if not only_one_class else float('nan'),
        "acc":          float((y_true == y_pred).mean()),
    }


# =========================
# TRAIN / TEST (FIXED: metriche calcolate sull'intero set)
# =========================

def train(model, loader, optimizer, task_names):
    model.train()
    device = next(model.parameters()).device
    total_loss = 0.0

    y_true_all = {t: [] for t in task_names}
    y_pred_all = {t: [] for t in task_names}
    y_prob_all = {t: [] for t in task_names}

    for data in loader:
        data = data.to(device)
        data.x         = data.x.float()
        data.edge_attr = data.edge_attr.float()
        data.y         = data.y.float()
        data.mask      = data.mask.float()

        optimizer.zero_grad()
        out  = model(data)
        loss = weighted_binary_crossentropy(data.y, out, data.mask, task_names)
        total_loss += loss.item() / len(loader)
        loss.backward()
        optimizer.step()

        pred = (out >= 0.5).float()

        with torch.no_grad():
            for t, task in enumerate(task_names):
                task_mask = data.mask[:, t].bool()
                if task_mask.sum() == 0:
                    continue
                y_true_all[task].extend(data.y[task_mask, t].cpu().tolist())
                y_pred_all[task].extend(pred[task_mask, t].cpu().tolist())
                y_prob_all[task].extend(out[task_mask, t].detach().cpu().tolist())

    metrics = {task: _compute_task_metrics(y_true_all[task],
                                            y_pred_all[task],
                                            y_prob_all[task])
               for task in task_names}

    return model, total_loss, metrics


@torch.no_grad()
def test(model, loader, task_names):
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0

    y_true_all = {t: [] for t in task_names}
    y_pred_all = {t: [] for t in task_names}
    y_prob_all = {t: [] for t in task_names}

    for data in loader:
        data = data.to(device)
        data.x         = data.x.float()
        data.edge_attr = data.edge_attr.float()
        data.y         = data.y.float()
        data.mask      = data.mask.float()

        out  = model(data)
        loss = weighted_binary_crossentropy(data.y, out, data.mask, task_names)
        total_loss += loss.item() / len(loader)

        pred = (out >= 0.5).float()

        for t, task in enumerate(task_names):
            task_mask = data.mask[:, t].bool()
            if task_mask.sum() == 0:
                continue
            y_true_all[task].extend(data.y[task_mask, t].cpu().tolist())
            y_pred_all[task].extend(pred[task_mask, t].cpu().tolist())
            y_prob_all[task].extend(out[task_mask, t].cpu().tolist())

    metrics = {task: _compute_task_metrics(y_true_all[task],
                                            y_pred_all[task],
                                            y_prob_all[task])
               for task in task_names}

    return total_loss, metrics


def accuracy(pred_y, y):
    return ((pred_y == y).sum() / len(y)).item()


def calculate_the_average(x):
    return np.average(x)
