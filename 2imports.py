from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import deepcopy
from graphnet.constants import GRAPHNET_ROOT_DIR
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.parquet import ParquetDataset
from graphnet.data.utilities.string_selection_resolver import StringSelectionResolver
from graphnet.models import Model
from graphnet.models import StandardModel
from graphnet.models.coarsening import Coarsening
from graphnet.models.components.layers import DynEdgeConv
from graphnet.models.detector.detector import Detector
from graphnet.models.detector.icecube import IceCubeKaggle
from graphnet.models.gnn.gnn import GNN
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.model import Model
from graphnet.models.task import Task
from graphnet.models.task.reconstruction import DirectionReconstructionWithKappa, ZenithReconstructionWithKappa, AzimuthReconstructionWithKappa
from graphnet.models.utils import calculate_distance_matrix
from graphnet.models.utils import calculate_xyzt_homophily
from graphnet.training.callbacks import ProgressBar, PiecewiseLinearLR
from graphnet.training.labels import Direction
from graphnet.training.loss_functions import VonMisesFisher3DLoss, VonMisesFisher2DLoss, LossFunction, VonMisesFisherLoss
from graphnet.training.utils import make_dataloader
from graphnet.utilities.config import Configurable, DatasetConfig, save_dataset_config
from graphnet.utilities.config import save_model_config
from graphnet.utilities.logging import LoggerMixin
from graphnet.utilities.logging import get_logger
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks import GradientAccumulationScheduler
from pytorch_lightning.profiler import PyTorchProfiler
from torch import Tensor
from torch import Tensor, LongTensor
from torch import nn
from torch.functional import Tensor
from torch.nn import ModuleList
from torch.nn.modules import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules.normalization import LayerNorm
from torch.optim import Adam
from torch.optim.adam import Adam
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.pool import knn_graph
from torch_geometric.typing import Adj
from torch_geometric.typing import Adj, OptTensor, PairOptTensor, PairTensor
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_max, scatter_mean, scatter_min, scatter_sum
from tqdm import tqdm
from typing import cast, Any, Callable, Optional, Sequence, Union, Dict, List, Optional, Union, Tuple
import gc
import graphnet
import numpy as np
import os
import pandas as pd
import random
import socket
import sys
import torch

try:
    from torch_cluster import knn
except ImportError:
    knn = None