import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GINConv,BatchNorm
from torch_geometric.data import Data
from torch_geometric.utils import from_rdmol
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from sklearn.metrics import f1_score, confusion_matrix, precision_score, average_precision_score, fbeta_score, matthews_corrcoef
from torch_geometric.loader import DataLoader
import torch
from torch.optim import Adam
from sklearn.metrics import matthews_corrcoef

### BATCH NORMALIZATION SOLO SE USO CDK E XTB DESCRITTORI NUMERICI

## Qui la batch normalization viene rimossa 

n_0, n_1 = None, None

def set_class_frequencies(n0, n1):
    global n_0, n_1
    n_0 = n0
    n_1 = n1
    print(f" Frequenze impostate: n_0 = {n_0} | n_1 = {n_1}")

class GINModel_cdk(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout, cdk_dim, out_channels=1):
        super(GINModel_cdk, self).__init__()
        self.convs = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.dropout = dropout

        input_dim = in_channels + cdk_dim  

        mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.convs.append(GINConv(mlp))
        self.batch_norm.append(BatchNorm(hidden_channels))  

        for _ in range(num_layers - 2):
            mlp_i = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            self.convs.append(GINConv(mlp_i))
            self.batch_norm.append(BatchNorm(hidden_channels))  

        # Layer finale senza cdk_dim
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        cdk_desc = data.cdk_desc

        x = torch.cat([x, cdk_desc], dim=-1)

        for conv, bn in zip(self.convs, self.batch_norm):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training) 

        x = self.lin(x)
        return torch.sigmoid(x).squeeze()


class GATC_cdk(nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_channels, num_layers, dropout, cdk_dim, out_channels=1):
        super(GATC_cdk, self).__init__()
        self.dropout = dropout
        self.heads = 4  
        self.out_heads = 1  
        self.convs = nn.ModuleList()
        self.batch_norm = nn.ModuleList()  

        input_dim = in_channels + cdk_dim  # 

        # Primo livello GAT
        self.convs.append(GATConv(input_dim, hidden_channels, heads=self.heads, edge_dim=edge_dim)) 
        self.batch_norm.append(BatchNorm(hidden_channels * self.heads))

        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * self.heads, hidden_channels, heads=self.heads, edge_dim=edge_dim))
            self.batch_norm.append(BatchNorm(hidden_channels * self.heads))

        self.convs.append(GATConv(hidden_channels * self.heads, hidden_channels, heads=self.out_heads, edge_dim=edge_dim))
        self.batch_norm.append(BatchNorm(hidden_channels * self.out_heads))

        # Layer finale senza cdk_dim
        self.lin = nn.Linear(hidden_channels * self.out_heads, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        cdk_desc = data.cdk_desc


        x = torch.cat([x, cdk_desc], dim=-1)

        for conv, bn in zip(self.convs, self.batch_norm):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lin(x)
        return torch.sigmoid(x).squeeze()

        
class GCNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout, cdk_dim, out_channels=1):
        super(GCNModel, self).__init__()
        self.convs = nn.ModuleList()
        self.batch_norm = torch.nn.ModuleList()  
        self.dropout = dropout
        
        input_dim = in_channels + cdk_dim  #
        
        self.convs.append(GCNConv(input_dim, hidden_channels))
        self.batch_norm.append(BatchNorm(hidden_channels))  

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norm.append(BatchNorm(hidden_channels))  
    
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.batch_norm.append(BatchNorm(hidden_channels))  
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        cdk_desc = data.cdk_desc
        
        x = torch.cat([x, cdk_desc], dim=-1)
        
        for conv, bn in zip(self.convs, self.batch_norm):
            x = conv(x, edge_index)
            x = bn(x)  
            x = F.elu(x)  
            x = F.dropout(x, p=self.dropout, training=self.training)  
        
        
        x = self.lin(x)
        return torch.sigmoid(x).squeeze()  


class GATC(nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_channels, num_layers, dropout, out_channels=1):
        super(GATC, self).__init__()
        self.dropout = dropout
        self.heads = 4  
        self.out_heads = 1  
        self.convs = nn.ModuleList() 

        # Primo livello GAT
        self.convs.append(GATConv(in_channels, hidden_channels, heads=self.heads, edge_dim=edge_dim)) 

        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * self.heads, hidden_channels, heads=self.heads, edge_dim=edge_dim))

        self.convs.append(GATConv(hidden_channels * self.heads, hidden_channels, heads=self.out_heads, edge_dim=edge_dim))
        self.lin = nn.Linear(hidden_channels * self.out_heads , out_channels)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.elu(x)  
            x = F.dropout(x, p=self.dropout, training=self.training) 
        
        x = self.lin(x)
        return torch.sigmoid(x).squeeze()  

class GINModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout, out_channels=1):
        super(GINModel, self).__init__()
        self.convs = nn.ModuleList()
        self.dropout = dropout
        
        mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.convs.append(GINConv(mlp))

        for _ in range(num_layers - 2):
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
            x = F.elu(x)  
            x = F.dropout(x, p=self.dropout, training=self.training) 
                
        x = self.lin(x)
        return torch.sigmoid(x).squeeze()
        

class GCNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout, out_channels=1):
        super(GCNModel, self).__init__()
        self.convs = nn.ModuleList()
        self.dropout = dropout
        
        self.convs.append(GCNConv(in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
     
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv in self.convs:
            x = conv(x, edge_index)  
            x = F.elu(x)  
            x = F.dropout(x, p=self.dropout, training=self.training)  
        
        x = self.lin(x)
        return torch.sigmoid(x).squeeze()

# definisco il modello e importo   
def weighted_binary_crossentropy(y_true, y_pred):
    weight_0 = (n_0 + n_1) / (2 * n_0)
    weight_1 = (n_0 + n_1) / (2 * n_1)
    bce_loss = F.binary_cross_entropy(y_pred, y_true, reduction='none')
    weights = y_true * weight_1 + (1 - y_true) * weight_0
    return (bce_loss * weights).mean()


def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    acc = 0
    train_precision = 0
    train_sensitivity = 0
    train_specificity = 0
    train_balanced_acc = 0
    train_fbeta = 0
    train_auprc = 0
    train_mcc = 0
    
    for data in loader:
        data.x = data.x.float()
        data.edge_attr = data.edge_attr.float()
        data.y = data.y.float()
        
        optimizer.zero_grad()
        out = model(data)
        loss = weighted_binary_crossentropy(data.y, out)
        total_loss += loss.item() / len(loader)
        
        pred = (out >= 0.5).float()
        acc += accuracy(pred, data.y) / len(loader)
        
        tn, fp, fn, tp = confusion_matrix(data.y.tolist(), pred.tolist(), labels=[0,1]).ravel()
        sensitivity_batch = tp / (tp + fn) if (tp + fn) > 0 else 0  
        specificity_batch = tn / (tn + fp) if (tn + fp) > 0 else 0  
        balanced_acc_batch = (sensitivity_batch + specificity_batch) / 2  
        precision_batch = precision_score(data.y.tolist(), pred.tolist(), zero_division=0)
        fbeta_batch = fbeta_score(data.y.tolist(), pred.tolist(), beta=0.5, zero_division=0)
        auprc_batch = average_precision_score(data.y.tolist(), out.tolist())
        matthews_corrcoef_batch = matthews_corrcoef(data.y.tolist(), pred.tolist())

        train_sensitivity += sensitivity_batch / len(loader)
        train_specificity += specificity_batch / len(loader)
        train_balanced_acc += balanced_acc_batch / len(loader)
        train_precision += precision_batch / len(loader)
        train_fbeta += fbeta_batch / len(loader)
        train_auprc += auprc_batch / len(loader)
        train_mcc += matthews_corrcoef_batch / len(loader)

        loss.backward()
        optimizer.step()
    
    return model, total_loss, acc, train_sensitivity, train_specificity, train_balanced_acc, train_precision, train_fbeta, train_auprc, train_mcc

@torch.no_grad()
def test(model, loader):
    model.eval()
    total_loss = 0
    acc = 0
    test_precision = 0
    test_sensitivity = 0
    test_specificity = 0
    test_balanced_acc = 0
    test_fbeta = 0
    test_auprc = 0
    test_mcc = 0

    for data in loader:
        data.x = data.x.float()
        data.edge_attr = data.edge_attr.float()
        data.y = data.y.float()
        
        out = model(data)
        loss = weighted_binary_crossentropy(data.y, out)
        total_loss += loss.item() / len(loader)
        
        pred = (out >= 0.5).float()
        acc += accuracy(pred, data.y) / len(loader)
        
        tn, fp, fn, tp = confusion_matrix(data.y.tolist(), pred.tolist(), labels=[0,1]).ravel()
        sensitivity_batch = tp / (tp + fn) if (tp + fn) > 0 else 0  
        specificity_batch = tn / (tn + fp) if (tn + fp) > 0 else 0  
        balanced_acc_batch = (sensitivity_batch + specificity_batch) / 2  
        precision_batch = precision_score(data.y.tolist(), pred.tolist(), zero_division=0)
        fbeta_batch = fbeta_score(data.y.tolist(), pred.tolist(), beta=0.5, zero_division=0)
        auprc_batch = average_precision_score(data.y.tolist(), out.tolist())
        matthews_corrcoef_batch = matthews_corrcoef(data.y.tolist(), pred.tolist())

        test_sensitivity += sensitivity_batch / len(loader)
        test_specificity += specificity_batch / len(loader)
        test_balanced_acc += balanced_acc_batch / len(loader)
        test_precision += precision_batch / len(loader)
        test_fbeta += fbeta_batch / len(loader)
        test_auprc += auprc_batch / len(loader)
        test_mcc += matthews_corrcoef_batch / len(loader)

    return total_loss, acc, test_sensitivity, test_specificity, test_balanced_acc, test_precision, test_fbeta, test_auprc, test_mcc

def accuracy(pred_y, y):
    return ((pred_y == y).sum() / len(y)).item()

def calculate_the_average(x):
    return np.average(x)      
