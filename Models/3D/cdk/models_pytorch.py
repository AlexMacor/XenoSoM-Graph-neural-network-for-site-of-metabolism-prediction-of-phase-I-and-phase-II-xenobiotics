import os
import os.path as osp
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import from_rdmol
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from sklearn.metrics import f1_score, confusion_matrix, precision_score, average_precision_score, fbeta_score
from sklearn.metrics import matthews_corrcoef
from torch_geometric.loader import DataLoader
import torch
from torch.optim import Adam
from functools import partial
from math import pi as PI
from math import sqrt
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import Embedding, Linear

from torch_geometric.data import Dataset, download_url
from torch_geometric.nn import radius_graph
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import OptTensor, SparseTensor
from torch_geometric.utils import scatter

from modules.envelope import Envelope
from modules.residual_layer import ResidualLayer
from modules.bessel_basis_layer import BesselBasisLayer
from modules.spherical_basis_layer import SphericalBasisLayer
from modules.embedding_block import EmbeddingBlock
from modules.interaction_pp_block import InteractionPPBlock
from modules.interaction_block import InteractionBlock
from modules.output_pp_block import OutputPPBlock
from modules.output_block import OutputBlock
from modules.triplets import triplets

device = torch.device('cpu')

n_0, n_1 = None, None

def set_class_frequencies(n0, n1):
    global n_0, n_1
    n_0 = n0
    n_1 = n1
    #print(f" Frequenze impostate: n_0 = {n_0} | n_1 = {n_1}")


class DimeNet(torch.nn.Module):
    r"""The directional message passing neural network (DimeNet) from the
    `"Directional Message Passing for Molecular Graphs"
    <https://arxiv.org/abs/2003.03123>`_ paper.
    DimeNet transforms messages based on the angle between them in a
    rotation-equivariant fashion.

    .. note::

        For an example of using a pretrained DimeNet variant, see
        `examples/qm9_pretrained_dimenet.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        qm9_pretrained_dimenet.py>`_.

    Args:
        hidden_channels (int): Hidden embedding size.
        out_channels (int): Size of each output sample.
        num_blocks (int): Number of building blocks.
        num_bilinear (int): Size of the bilinear layer tensor.
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
        cutoff (float, optional): Cutoff distance for interatomic
            interactions. (default: :obj:`5.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: :obj:`32`)
        envelope_exponent (int, optional): Shape of the smooth cutoff.
            (default: :obj:`5`)
        num_before_skip (int, optional): Number of residual layers in the
            interaction blocks before the skip connection. (default: :obj:`1`)
        num_after_skip (int, optional): Number of residual layers in the
            interaction blocks after the skip connection. (default: :obj:`2`)
        num_output_layers (int, optional): Number of linear layers for the
            output blocks. (default: :obj:`3`)
        act (str or Callable, optional): The activation function.
            (default: :obj:`"swish"`)
        output_initializer (str, optional): The initialization method for the
            output layer (:obj:`"zeros"`, :obj:`"glorot_orthogonal"`).
            (default: :obj:`"zeros"`)
    """

    url = ('https://github.com/klicperajo/dimenet/raw/master/pretrained/'
           'dimenet')

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        num_blocks: int,
        num_bilinear: int,
        num_spherical: int,
        num_radial: int,
        cdk_dim: int = 0,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
        act: Union[str, Callable] = 'swish',
        output_initializer: str = 'zeros',
    ):
        super().__init__()

        if num_spherical < 2:
            raise ValueError("'num_spherical' should be greater than 1")

        act = activation_resolver(act)
        
        self.cdk_dim = cdk_dim
        self.input_dim = hidden_channels + cdk_dim
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.num_blocks = num_blocks

        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, cutoff,
                                       envelope_exponent)

        self.emb = EmbeddingBlock(num_radial, hidden_channels, act, cdk_dim=cdk_dim)

        self.output_blocks = torch.nn.ModuleList([
            OutputBlock(
                num_radial,
                hidden_channels,
                out_channels,
                num_output_layers,
                act,
                output_initializer,
            ) for _ in range(num_blocks + 1)
        ])

        self.interaction_blocks = torch.nn.ModuleList([
            InteractionBlock(
                hidden_channels,
                num_bilinear,
                num_spherical,
                num_radial,
                num_before_skip,
                num_after_skip,
                act,
            ) for _ in range(num_blocks)
        ])
        
        self.node_classifier = torch.nn.Linear(out_channels, 1)
        
        
    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.rbf.reset_parameters()
        self.emb.reset_parameters()
        for out in self.output_blocks:
            out.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()

        
    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: OptTensor = None,
        data: Optional[Data] = None
    ) -> Tensor:
        r"""Forward pass.

        Args:
            z (torch.Tensor): Atomic number of each atom with shape
                :obj:`[num_atoms]`.
            pos (torch.Tensor): Coordinates of each atom with shape
                :obj:`[num_atoms, 3]`.
            batch (torch.Tensor, optional): Batch indices assigning each atom
                to a separate molecule with shape :obj:`[num_atoms]`.
                (default: :obj:`None`)
        """
        
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)

        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
            edge_index, num_nodes=z.size(0))

        # Calculate distances.
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        # Calculate angles.
        if isinstance(self, DimeNetPP):
            pos_jk, pos_ij = pos[idx_j] - pos[idx_k], pos[idx_i] - pos[idx_j]
            a = (pos_ij * pos_jk).sum(dim=-1)
            b = torch.cross(pos_ij, pos_jk, dim=1).norm(dim=-1)
        elif isinstance(self, DimeNet):
            pos_ji, pos_ki = pos[idx_j] - pos[idx_i], pos[idx_k] - pos[idx_i]
            a = (pos_ji * pos_ki).sum(dim=-1)
            b = torch.cross(pos_ji, pos_ki, dim=1).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        # Embedding block.

        x = self.emb(z, rbf, i, j, data.cdk_desc)
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))

        # Interaction blocks.
        for interaction_block, output_block in zip(self.interaction_blocks,
                                                   self.output_blocks[1:]):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P = P + output_block(x, rbf, i, num_nodes=pos.size(0))

        logits = self.node_classifier(P)
        return torch.sigmoid(logits).squeeze(-1)
        
        #if batch is None:
        #    return P.sum(dim=0)
        #else:
        #    return scatter(P, batch, dim=0, reduce='sum')



class DimeNetPP(DimeNet):
    r"""The DimeNet++ from the
    `"Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules"
    <https://arxiv.org/abs/2011.14115>`_ paper.

    :class:`DimeNetPlusPlus` is an upgrade to the :class:`DimeNet` model 
    with 8x faster and ~10% more accurate results.


    Args:
        hidden_channels (int): Hidden embedding size.
        out_channels (int): Size of each output sample.
        num_blocks (int): Number of building blocks.
        int_emb_size (int): Size of embedding in the interaction block.
        basis_emb_size (int): Size of basis embedding in the interaction block.
        out_emb_channels (int): Size of embedding in the output block.
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
        cutoff (float, optional): Cutoff distance for interatomic interactions. (default: 5.0)
        max_num_neighbors (int, optional): Max neighbors per node within cutoff. (default: 32)
        envelope_exponent (int, optional): Shape of the smooth cutoff. (default: 5)
        num_before_skip (int, optional): Residual layers before skip. (default: 1)
        num_after_skip (int, optional): Residual layers after skip. (default: 2)
        num_output_layers (int, optional): Linear layers for output blocks. (default: 3)
        act (str or Callable, optional): Activation function. (default: "swish")
        output_initializer (str, optional): Output layer initialization ("zeros" or "glorot_orthogonal"). (default: "zeros")
    """

    url = ('https://raw.githubusercontent.com/gasteigerjo/dimenet/'
           'master/pretrained/dimenet_pp')

    def __init__(
        self,
        hidden_channels: int,
        num_blocks: int,
        int_emb_size: int,
        basis_emb_size: int,
        out_emb_channels: int,
        num_spherical: int,
        num_radial: int,
        out_channels: int=1,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
        act: Union[str, Callable] = 'swish',
        output_initializer: str = 'zeros',
        cdk_dim: int = 0,
    ):
        act = activation_resolver(act)
        
    
    
        # Call DimeNet constructor (inherits RBF, SBF, embedding)
        super().__init__(
            hidden_channels=hidden_channels,
            out_channels=out_emb_channels,
            num_blocks=num_blocks,
            num_bilinear=1,  # Placeholder, unused in DimeNet++
            num_spherical=num_spherical,
            num_radial=num_radial,
            cdk_dim=cdk_dim,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_output_layers=num_output_layers,
            act=act,
            output_initializer=output_initializer
        )
        
        
        
        # Replace original output and interaction blocks with upgraded versions
        self.output_blocks = torch.nn.ModuleList([
            OutputPPBlock(
                num_radial,
                hidden_channels,
                out_emb_channels,
                out_emb_channels,
                num_output_layers,
                act,
                output_initializer,
            ) for _ in range(num_blocks + 1)
        ])

        self.interaction_blocks = torch.nn.ModuleList([
            InteractionPPBlock(
                hidden_channels,
                int_emb_size,
                basis_emb_size,
                num_spherical,
                num_radial,
                num_before_skip,
                num_after_skip,
                act,
            ) for _ in range(num_blocks)
        ])
        
        self.node_classifier = torch.nn.Linear(out_emb_channels, 1)
        self.reset_parameters()

    def forward(self, z, pos, batch=None, data=None):    
        return super().forward(z, pos, batch=batch, data=data)

   
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
        #data.x = data.x.float()
        #data.edge_attr = data.edge_attr.float()
        data.y = data.y.float()
        
        optimizer.zero_grad()
        #out = model(data)
        out = model(data.z, data.pos, batch=data.batch, data=data)
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
        #data.x = data.x.float()
        #data.edge_attr = data.edge_attr.float()
        data.y = data.y.float()
        
        #out = model(data)
        out = model(data.z, data.pos, batch=data.batch, data=data)
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
