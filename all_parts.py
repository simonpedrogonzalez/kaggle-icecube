from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import deepcopy
from graphnet.constants import GRAPHNET_ROOT_DIR
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.utilities.string_selection_resolver import StringSelectionResolver
from graphnet.models import Model
from graphnet.models import StandardModel
from graphnet.models.coarsening import Coarsening
from graphnet.models.components.layers import DynEdgeConv
from graphnet.models.detector.detector import Detector
from graphnet.models.detector.icecube import IceCubeKaggle
from graphnet.models.gnn.gnn import GNN
# from graphnet.models.graph_builders import KNNGraphBuilder
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
from graphnet.utilities.logging import Logger
# from graphnet.utilities.logging import get_logger
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks import GradientAccumulationScheduler
# from pytorch_lightning.profiler import PyTorchProfiler
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

# create symlink input folder to
# /media/simon/Extrene SSD/input
try:
    os.symlink("/media/simon/Extreme SSD/input", "./input")
    print(f"Symlink created")
except FileExistsError:
    print("Symlink already exists!")
except Exception as e:
    print(f"Error: {e}")



class ColumnMissingException(Exception):
    """Exception to indicate a missing column in a dataset."""


class Dataset2(torch.utils.data.Dataset, Configurable, Logger, ABC):
    """Base Dataset class for reading from any intermediate file format."""

    # Class method(s)
    @classmethod
    def from_config(  # type: ignore[override]
        cls,
        source: Union[DatasetConfig, str],
    ) -> Union[
        "Dataset",
        ConcatDataset,
        Dict[str, "Dataset"],
        Dict[str, ConcatDataset],
    ]:
        """Construct `Dataset` instance from `source` configuration."""
        if isinstance(source, str):
            source = DatasetConfig.load(source)

        assert isinstance(source, DatasetConfig), (
            f"Argument `source` of type ({type(source)}) is not a "
            "`DatasetConfig"
        )

        # Parse set of `selection``.
        if isinstance(source.selection, dict):
            return cls._construct_datasets_from_dict(source)
        elif (
            isinstance(source.selection, list)
            and len(source.selection)
            and isinstance(source.selection[0], str)
        ):
            return cls._construct_dataset_from_list_of_strings(source)

        return source._dataset_class(**source.dict())

    @classmethod
    def concatenate(
        cls,
        datasets: List["Dataset"],
    ) -> ConcatDataset:
        """Concatenate multiple `Dataset`s into one instance."""
        return ConcatDataset(datasets)

    @classmethod
    def _construct_datasets_from_dict(
        cls, config: DatasetConfig
    ) -> Dict[str, "Dataset"]:
        """Construct `Dataset` for each entry in dict `self.selection`."""
        assert isinstance(config.selection, dict)
        datasets: Dict[str, "Dataset"] = {}
        selections: Dict[str, Union[str, List]] = deepcopy(config.selection)
        for key, selection in selections.items():
            config.selection = selection
            dataset = Dataset.from_config(config)
            assert isinstance(dataset, (Dataset, ConcatDataset))
            datasets[key] = dataset

        # Reset `selections`.
        config.selection = selections

        return datasets

    @classmethod
    def _construct_dataset_from_list_of_strings(
        cls, config: DatasetConfig
    ) -> "Dataset":
        """Construct `Dataset` for each entry in list `self.selection`."""
        assert isinstance(config.selection, list)
        datasets: List["Dataset"] = []
        selections: List[str] = deepcopy(cast(List[str], config.selection))
        for selection in selections:
            config.selection = selection
            dataset = Dataset.from_config(config)
            assert isinstance(dataset, Dataset)
            datasets.append(dataset)

        # Reset `selections`.
        config.selection = selections

        return cls.concatenate(datasets)

    @classmethod
    def _resolve_graphnet_paths(
        cls, path: Union[str, List[str]]
    ) -> Union[str, List[str]]:
        if isinstance(path, list):
            return [cast(str, cls._resolve_graphnet_paths(p)) for p in path]

        assert isinstance(path, str)
        return (
            path.replace("$graphnet", GRAPHNET_ROOT_DIR)
            .replace("$GRAPHNET", GRAPHNET_ROOT_DIR)
            .replace("${graphnet}", GRAPHNET_ROOT_DIR)
            .replace("${GRAPHNET}", GRAPHNET_ROOT_DIR)
        )

    @save_dataset_config
    def __init__(
        self,
        path: Union[str, List[str]],
        pulsemaps: Union[str, List[str]],
        features: List[str],
        truth: List[str],
        *,
        node_truth: Optional[List[str]] = None,
        index_column: str = "event_no",
        truth_table: str = "truth",
        node_truth_table: Optional[str] = None,
        string_selection: Optional[List[int]] = None,
        selection: Optional[Union[str, List[int], List[List[int]]]] = None,
        dtype: torch.dtype = torch.float32,
        loss_weight_table: Optional[str] = None,
        loss_weight_column: Optional[str] = None,
        loss_weight_default_value: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        """Construct Dataset.

        Args:
            path: Path to the file(s) from which this `Dataset` should read.
            pulsemaps: Name(s) of the pulse map series that should be used to
                construct the nodes on the individual graph objects, and their
                features. Multiple pulse series maps can be used, e.g., when
                different DOM types are stored in different maps.
            features: List of columns in the input files that should be used as
                node features on the graph objects.
            truth: List of event-level columns in the input files that should
                be used added as attributes on the  graph objects.
            node_truth: List of node-level columns in the input files that
                should be used added as attributes on the graph objects.
            index_column: Name of the column in the input files that contains
                unique indicies to identify and map events across tables.
            truth_table: Name of the table containing event-level truth
                information.
            node_truth_table: Name of the table containing node-level truth
                information.
            string_selection: Subset of strings for which data should be read
                and used to construct graph objects. Defaults to None, meaning
                all strings for which data exists are used.
            selection: The events that should be read. This can be given either
                as list of indicies (in `index_column`); or a string-based
                selection used to query the `Dataset` for events passing the
                selection. Defaults to None, meaning that all events in the
                input files are read.
            dtype: Type of the feature tensor on the graph objects returned.
            loss_weight_table: Name of the table containing per-event loss
                weights.
            loss_weight_column: Name of the column in `loss_weight_table`
                containing per-event loss weights. This is also the name of the
                corresponding attribute assigned to the graph object.
            loss_weight_default_value: Default per-event loss weight.
                NOTE: This default value is only applied when
                `loss_weight_table` and `loss_weight_column` are specified, and
                in this case to events with no value in the corresponding
                table/column. That is, if no per-event loss weight table/column
                is provided, this value is ignored. Defaults to None.
            seed: Random number generator seed, used for selecting a random
                subset of events when resolving a string-based selection (e.g.,
                `"10000 random events ~ event_no % 5 > 0"` or `"20% random
                events ~ event_no % 5 > 0"`).
        """
        # Check(s)
        if isinstance(pulsemaps, str):
            pulsemaps = [pulsemaps]

        assert isinstance(features, (list, tuple))
        assert isinstance(truth, (list, tuple))

        self._logger = Logger(class_name=self.__class__.__name__)

        # Resolve reference to `$GRAPHNET` in path(s)
        path = self._resolve_graphnet_paths(path)

        # Member variable(s)
        self._path = path
        self._selection = None
        self._pulsemaps = pulsemaps
        self._features = [index_column] + features
        self._truth = [index_column] + truth
        self._index_column = index_column
        self._truth_table = truth_table
        self._loss_weight_default_value = loss_weight_default_value

        if node_truth is not None:
            assert isinstance(node_truth_table, str)
            if isinstance(node_truth, str):
                node_truth = [node_truth]

        self._node_truth = node_truth
        self._node_truth_table = node_truth_table

        if string_selection is not None:
            self.warning(
                (
                    "String selection detected.\n "
                    f"Accepted strings: {string_selection}\n "
                    "All other strings are ignored!"
                )
            )
            if isinstance(string_selection, int):
                string_selection = [string_selection]

        self._string_selection = string_selection

        self._selection = None
        if self._string_selection:
            self._selection = f"string in {str(tuple(self._string_selection))}"

        self._loss_weight_column = loss_weight_column
        self._loss_weight_table = loss_weight_table
        if (self._loss_weight_table is None) and (
            self._loss_weight_column is not None
        ):
            self.warning("Error: no loss weight table specified")
            assert isinstance(self._loss_weight_table, str)
        if (self._loss_weight_table is not None) and (
            self._loss_weight_column is None
        ):
            self.warning("Error: no loss weight column specified")
            assert isinstance(self._loss_weight_column, str)

        self._dtype = dtype

        self._label_fns: Dict[str, Callable[[Data], Any]] = {}

        self._string_selection_resolver = StringSelectionResolver(
            self,
            index_column=index_column,
            seed=seed,
        )

        # Implementation-specific initialisation.
        self._init()

        # Set unique indices
        self._indices: Union[List[int], List[List[int]]]
        if selection is None:
            self._indices = self._get_all_indices()
        elif isinstance(selection, str):
            self._indices = self._resolve_string_selection_to_indices(
                selection
            )
        else:
            self._indices = selection

        # Purely internal member variables
        self._missing_variables: Dict[str, List[str]] = {}
        self._remove_missing_columns()

        # Implementation-specific post-init code.
        self._post_init()

        # Base class constructor
        super().__init__()

    # Properties
    @property
    def path(self) -> Union[str, List[str]]:
        """Path to the file(s) from which this `Dataset` reads."""
        return self._path

    @property
    def truth_table(self) -> str:
        """Name of the table containing event-level truth information."""
        return self._truth_table

    # Abstract method(s)
    @abstractmethod
    def _init(self) -> None:
        """Set internal representation needed to read data from input file."""

    def _post_init(self) -> None:
        """Implemenation-specific code to be run after the main constructor."""

    @abstractmethod
    def _get_all_indices(self) -> List[int]:
        """Return a list of all available values in `self._index_column`."""

    @abstractmethod
    def _get_event_index(
        self, sequential_index: Optional[int]
    ) -> Optional[int]:
        """Return a the event index corresponding to a `sequential_index`."""

    @abstractmethod
    def query_table(
        self,
        table: str,
        columns: Union[List[str], str],
        sequential_index: Optional[int] = None,
        selection: Optional[str] = None,
    ) -> List[Tuple[Any, ...]]:
        """Query a table at a specific index, optionally with some selection.

        Args:
            table: Table to be queried.
            columns: Columns to read out.
            sequential_index: Sequentially numbered index
                (i.e. in [0,len(self))) of the event to query. This _may_
                differ from the indexation used in `self._indices`. If no value
                is provided, the entire column is returned.
            selection: Selection to be imposed before reading out data.
                Defaults to None.

        Returns:
            List of tuples containing the values in `columns`. If the `table`
                contains only scalar data for `columns`, a list of length 1 is
                returned

        Raises:
            ColumnMissingException: If one or more element in `columns` is not
                present in `table`.
        """

    # Public method(s)
    def add_label(self, key: str, fn: Callable[[Data], Any]) -> None:
        """Add custom graph label define using function `fn`."""
        assert (
            key not in self._label_fns
        ), f"A custom label {key} has already been defined."
        self._label_fns[key] = fn

    def __len__(self) -> int:
        """Return number of graphs in `Dataset`."""
        return len(self._indices)

    def __getitem__(self, sequential_index: int) -> Data:
        """Return graph `Data` object at `index`."""
        if not (0 <= sequential_index < len(self)):
            raise IndexError(
                f"Index {sequential_index} not in range [0, {len(self) - 1}]"
            )
        #import pdb;pdb.set_trace()
        features, truth, node_truth, loss_weight = self._query(
            sequential_index
        )
        graph = self._create_graph(features, truth, node_truth, loss_weight)
        return graph

    # Internal method(s)
    def _resolve_string_selection_to_indices(
        self, selection: str
    ) -> List[int]:
        """Resolve selection as string to list of indicies.

        Selections are expected to have pandas.DataFrame.query-compatible
        syntax, e.g., ``` "event_no % 5 > 0" ``` Selections may also specify a
        fixed number of events to randomly sample, e.g., ``` "10000 random
        events ~ event_no % 5 > 0" "20% random events ~ event_no % 5 > 0" ```
        """
        return self._string_selection_resolver.resolve(selection)

    def _remove_missing_columns(self) -> None:
        """Remove columns that are not present in the input file.

        Columns are removed from `self._features` and `self._truth`.
        """
        # Check if table is completely empty
        if len(self) == 0:
            self.warning("Dataset is empty.")
            return

        # Find missing features
        missing_features_set = set(self._features)
        for pulsemap in self._pulsemaps:
            missing = self._check_missing_columns(self._features, pulsemap)
            missing_features_set = missing_features_set.intersection(missing)

        missing_features = list(missing_features_set)

        # Find missing truth variables
        missing_truth_variables = self._check_missing_columns(
            self._truth, self._truth_table
        )

        # Remove missing features
        if missing_features:
            self.warning(
                "Removing the following (missing) features: "
                + ", ".join(missing_features)
            )
            for missing_feature in missing_features:
                self._features.remove(missing_feature)

        # Remove missing truth variables
        if missing_truth_variables:
            self.warning(
                (
                    "Removing the following (missing) truth variables: "
                    + ", ".join(missing_truth_variables)
                )
            )
            for missing_truth_variable in missing_truth_variables:
                self._truth.remove(missing_truth_variable)

    def _check_missing_columns(
        self,
        columns: List[str],
        table: str,
    ) -> List[str]:
        """Return a list missing columns in `table`."""
        for column in columns:
            try:
                self.query_table(table, [column], 0)
            except ColumnMissingException:
                if table not in self._missing_variables:
                    self._missing_variables[table] = []
                self._missing_variables[table].append(column)
            except IndexError:
                self.warning(f"Dataset contains no entries for {column}")
            except:
                if table not in self._missing_variables:
                    self._missing_variables[table] = []
                self._missing_variables[table].append(column)

        return self._missing_variables.get(table, [])

    def _query(
        self, sequential_index: int
    ) -> Tuple[
        List[Tuple[float, ...]],
        Tuple[Any, ...],
        Optional[List[Tuple[Any, ...]]],
        Optional[float],
    ]:
        """Query file for event features and truth information.

        The returned lists have lengths correspondings to the number of pulses
        in the event. Their constituent tuples have lengths corresponding to
        the number of features/attributes in each output

        Args:
            sequential_index: Sequentially numbered index
                (i.e. in [0,len(self))) of the event to query. This _may_
                differ from the indexation used in `self._indices`.

        Returns:
            Tuple containing pulse-level event features; event-level truth
                information; pulse-level truth information; and event-level
                loss weights, respectively.
        """
        #import pdb;pdb.set_trace()
        features = []
        for pulsemap in self._pulsemaps:
            features_pulsemap = self.query_table(
                pulsemap, self._features, sequential_index, self._selection
            )
            features.extend(features_pulsemap)

        truth: Tuple[Any, ...] = self.query_table(
            self._truth_table, self._truth, sequential_index
        )# [0] # updated
        if self._node_truth:
            assert self._node_truth_table is not None
            node_truth = self.query_table(
                self._node_truth_table,
                self._node_truth,
                sequential_index,
                self._selection,
            )
        else:
            node_truth = None

        loss_weight: Optional[float] = None  # Default
        if self._loss_weight_column is not None:
            assert self._loss_weight_table is not None
            loss_weight_list = self.query_table(
                self._loss_weight_table,
                self._loss_weight_column,
                sequential_index,
            )
            if len(loss_weight_list):
                loss_weight = loss_weight_list[0][0]
            else:
                loss_weight = -1.0

        return features, truth, node_truth, loss_weight

    def _create_graph(
        self,
        features: List[Tuple[float, ...]],
        truth: Tuple[Any, ...],
        node_truth: Optional[List[Tuple[Any, ...]]] = None,
        loss_weight: Optional[float] = None,
    ) -> Data:
        """Create Pytorch Data (i.e. graph) object.

        No preprocessing is performed at this stage, just as no node adjancency
        is imposed. This means that the `edge_attr` and `edge_weight`
        attributes are not set.

        Args:
            features: List of tuples, containing event features.
            truth: List of tuples, containing truth information.
            node_truth: List of tuples, containing node-level truth.
            loss_weight: A weight associated with the event for weighing the
                loss.

        Returns:
            Graph object.
        """
        # Convert nested list to simple dict
        truth_dict = {
            key: truth[index] for index, key in enumerate(self._truth)
        }

        # Define custom labels
        labels_dict = self._get_labels(truth_dict)

        # Convert nested list to simple dict
        if node_truth is not None:
            node_truth_array = np.asarray(node_truth)
            assert self._node_truth is not None
            node_truth_dict = {
                key: node_truth_array[:, index]
                for index, key in enumerate(self._node_truth)
            }

        # updated
        # Catch cases with no reconstructed pulses
        if len(features):
            data = np.asarray(features)[:, 1:]
        else:
            data = np.array([]).reshape((0, len(self._features) - 1))
        #data = features[:, 1:]

        # Construct graph data object
        x = torch.tensor(data.astype('float32'), dtype=self._dtype)  # pylint: disable=C0103
        n_pulses = torch.tensor(len(x), dtype=torch.int32)
        graph = Data(x=x, edge_index=None)
        graph.n_pulses = n_pulses
        graph.features = self._features[1:]

        # Add loss weight to graph.
        if loss_weight is not None and self._loss_weight_column is not None:
            # No loss weight was retrieved, i.e., it is missing for the current
            # event.
            if loss_weight < 0:
                if self._loss_weight_default_value is None:
                    raise ValueError(
                        "At least one event is missing an entry in "
                        f"{self._loss_weight_column} "
                        "but loss_weight_default_value is None."
                    )
                graph[self._loss_weight_column] = torch.tensor(
                    self._loss_weight_default_value, dtype=self._dtype
                ).reshape(-1, 1)
            else:
                graph[self._loss_weight_column] = torch.tensor(
                    loss_weight, dtype=self._dtype
                ).reshape(-1, 1)

        # Write attributes, either target labels, truth info or original
        # features.
        add_these_to_graph = [labels_dict, truth_dict]
        if node_truth is not None:
            add_these_to_graph.append(node_truth_dict)
        for write_dict in add_these_to_graph:
            for key, value in write_dict.items():
                try:
                    graph[key] = torch.tensor(value)
                except TypeError:
                    # Cannot convert `value` to Tensor due to its data type,
                    # e.g. `str`.
                    self.debug(
                        (
                            f"Could not assign `{key}` with type "
                            f"'{type(value).__name__}' as attribute to graph."
                        )
                    )

        # Additionally add original features as (static) attributes
        for index, feature in enumerate(graph.features):
            if feature not in ["x"]:
                graph[feature] = graph.x[:, index].detach()

        # Add custom labels to the graph
        for key, fn in self._label_fns.items():
            graph[key] = fn(graph)
        return graph

    def _get_labels(self, truth_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Return dictionary of  labels, to be added as graph attributes."""
        if "pid" in truth_dict.keys():
            abs_pid = abs(truth_dict["pid"])
            sim_type = truth_dict["sim_type"]

            labels_dict = {
                self._index_column: truth_dict[self._index_column],
                "muon": int(abs_pid == 13),
                "muon_stopped": int(truth_dict.get("stopped_muon") == 1),
                "noise": int((abs_pid == 1) & (sim_type != "data")),
                "neutrino": int(
                    (abs_pid != 13) & (abs_pid != 1)
                ),  # @TODO: `abs_pid in [12,14,16]`?
                "v_e": int(abs_pid == 12),
                "v_u": int(abs_pid == 14),
                "v_t": int(abs_pid == 16),
                "track": int(
                    (abs_pid == 14) & (truth_dict["interaction_type"] == 1)
                ),
                "dbang": self._get_dbang_label(truth_dict),
                "corsika": int(abs_pid > 20),
            }
        else:
            labels_dict = {
                self._index_column: truth_dict[self._index_column],
                "muon": -1,
                "muon_stopped": -1,
                "noise": -1,
                "neutrino": -1,
                "v_e": -1,
                "v_u": -1,
                "v_t": -1,
                "track": -1,
                "dbang": -1,
                "corsika": -1,
            }
        return labels_dict

    def _get_dbang_label(self, truth_dict: Dict[str, Any]) -> int:
        """Get label for double-bang classification."""
        try:
            label = int(truth_dict["dbang_decay_length"] > -1)
            return label
        except KeyError:
            return -1

class ParquetDataset2(Dataset2):
    """Pytorch dataset for reading from Parquet files."""
    
    def __init__(
        self,
        path: Union[str, List[str]],
        pulsemaps: Union[str, List[str]],
        features: List[str],
        truth: List[str],
        batch_ids = List[int],
        *,
        node_truth: Optional[List[str]] = None,
        index_column: str = "event_no",
        truth_table: str = "truth",
        node_truth_table: Optional[str] = None,
        string_selection: Optional[List[int]] = None,
        selection: Optional[Union[str, List[int], List[List[int]]]] = None,
        dtype: torch.dtype = torch.float32,
        loss_weight_table: Optional[str] = None,
        loss_weight_column: Optional[str] = None,
        loss_weight_default_value: Optional[float] = None,
        seed: Optional[int] = None,
        max_len:  Optional[int] = 0,
        max_pulse:  Optional[int] = 200,
        min_pulse:  Optional[int] = 0,
        
    ):
        self.batch_ids = batch_ids
        self.max_len = max_len
        self.max_pulse = max_pulse
        self.min_pulse = min_pulse
        self.this_batch_id = 0
        
        super().__init__(
        path=path,
        pulsemaps=pulsemaps,
        features=features,
        truth=truth,
        selection=selection,
        node_truth=node_truth,
        truth_table=truth_table,
        node_truth_table=node_truth_table,
        string_selection=string_selection,
        loss_weight_table=loss_weight_table,
        loss_weight_column=loss_weight_column,
        index_column=index_column,
        )
        
    def reset_epoch(self) -> None:
        self.this_batch_idx += 1
        if self.this_batch_idx >= len(self.batch_ids):
            self.this_batch_idx = 0
            
        if self.this_batch_id == self.batch_ids[self.this_batch_idx]:
            print('skip reset epoch ', self.this_batch_id, self.this_batch_idx)
            return
        else:
            self.this_batch_id = self.batch_ids[self.this_batch_idx]
            print('reset epoch to batch_id:', self.this_batch_id, self.this_batch_idx)

        #print('reading meta', f'input/train/meta_{self.this_batch_id}.parquet')
        meta = pd.read_parquet(f'{META_DIR}/meta_{self.this_batch_id}.parquet')
            
        if self.max_pulse > 0:
            pulse_count = meta.last_pulse_index - meta.first_pulse_index +1
            meta = meta[pulse_count<self.max_pulse].reset_index(drop=True)
        if self.min_pulse > 0:
            pulse_count = meta.last_pulse_index - meta.first_pulse_index +1
            meta = meta[pulse_count>=self.min_pulse].reset_index(drop=True)

            
        self.meta_name_index = {n:i for i,n in enumerate(meta.columns)}
        self.this_meta_arr = meta.values
        
        batch = pd.read_parquet(f"{BATCH_DIR}/batch_{self.this_batch_id}.parquet").reset_index()
        batch['x'] = batch['sensor_id'].map(self.sensor_geometry_dict['x'])
        batch['y'] = batch['sensor_id'].map(self.sensor_geometry_dict['y'])
        batch['z'] = batch['sensor_id'].map(self.sensor_geometry_dict['z'])
        self.batch_name_index = {n:i for i,n in enumerate(batch.columns)}
        self.this_batch = batch
        
    def _init(self) -> None:
        self.this_batch_idx = -1
        sensor_geometry = pd.read_csv(f'input/icecube-neutrinos-in-deep-ice/sensor_geometry.csv')
        self.sensor_geometry_dict = sensor_geometry.set_index('sensor_id').to_dict()

        self.reset_epoch()
        #print('done')
        
    def __len__(self) -> int:
        """Return number of graphs in `Dataset`."""
        if self.max_len > 0:
            return self.max_len
        else:
            return len(self.this_meta_arr)

    def _get_all_indices(self) -> List[int]:
        return range(len(self.this_meta_arr))

    def _get_event_index(sequential_index):
        return self.this_meta_arr.iloc[sequential_index,self.meta_name_index['event_id']]

    def query_table(
        self,
        table: str,
        columns: Union[List[str], str],
        sequential_index: Optional[int] = None,
        selection: Optional[str] = None,
    ) -> List[Tuple[Any, ...]]:
        if table == 'pulse_table':
            #columns = [self.batch_name_index[c] for c in columns]
            first_pulse_index = int(self.this_meta_arr[sequential_index,self.meta_name_index['first_pulse_index']])
            last_pulse_index = int(self.this_meta_arr[sequential_index,self.meta_name_index['last_pulse_index']])
            last_pulse_index = min(first_pulse_index+FORCE_MAX_PULSE, last_pulse_index)
            this_batch = self.this_batch[first_pulse_index:last_pulse_index+1]
            if ONLY_AUX_FALSE:
                this_batch = this_batch[this_batch.auxiliary == False]
            if len(this_batch)==0:
                new_sequential_index = np.random.randint(self.__len__())
                print('Warning: batch len is 0 for sequential_index, new_sequential_index', sequential_index, new_sequential_index)
                return self.query_table(table, columns, new_sequential_index, selection)
            return this_batch[columns].values
        else:
            columns = [self.meta_name_index[c] for c in columns]
            return self.this_meta_arr[sequential_index, columns]



def collate_fn(graphs: List[Data]) -> Batch:
    """Remove graphs with less than two DOM hits.

    Should not occur in "production.
    """
    graphs = [g for g in graphs if g.n_pulses > 1]
    return Batch.from_data_list(graphs)

# @TODO: Remove in favour of DataLoader{,.from_dataset_config}
def make_dataloader2(
    db: str,
    pulsemaps: Union[str, List[str]],
    features: List[str],
    truth: List[str],
    batch_ids = List[int],
    max_len = 0,
    max_pulse = 200,
    min_pulse = 200,
    *,
    batch_size: int,
    shuffle: bool,
    selection: Optional[List[int]] = None,
    num_workers: int = 10,
    persistent_workers: bool = False,
    node_truth: List[str] = None,
    truth_table: str = "truth",
    node_truth_table: Optional[str] = None,
    string_selection: List[int] = None,
    loss_weight_table: Optional[str] = None,
    loss_weight_column: Optional[str] = None,
    index_column: str = "event_no",
    labels: Optional[Dict[str, Callable]] = None,
) -> DataLoader:
    """Construct `DataLoader` instance."""
    # Check(s)
    if isinstance(pulsemaps, str):
        pulsemaps = [pulsemaps]

    dataset = ParquetDataset2(
        path=db,
        pulsemaps=pulsemaps,
        features=features,
        truth=truth,
        batch_ids=batch_ids,
        selection=selection,
        node_truth=node_truth,
        truth_table=truth_table,
        node_truth_table=node_truth_table,
        string_selection=string_selection,
        loss_weight_table=loss_weight_table,
        loss_weight_column=loss_weight_column,
        index_column=index_column,
        max_len=max_len,
        max_pulse=max_pulse,
        min_pulse=min_pulse,
    )

    # adds custom labels to dataset
    if isinstance(labels, dict):
        for label in labels.keys():
            dataset.add_label(key=label, fn=labels[label])

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        persistent_workers=persistent_workers,
        prefetch_factor=2,
    )

    return dataloader, dataset



class StandardModel2(Model):
    """Main class for standard models in graphnet.

    This class chains together the different elements of a complete GNN-based
    model (detector read-in, GNN architecture, and task-specific read-outs).
    """

    @save_model_config
    def __init__(
        self,
        *,
        detector: Detector,
        gnn: GNN,
        tasks: Union[Task, List[Task]],
        dataset: Dataset2,
        max_epochs: 0,
        coarsening: Optional[Coarsening] = None,
        optimizer_class: type = Adam,
        optimizer_kwargs: Optional[Dict] = None,
        scheduler_class: Optional[type] = None,
        scheduler_kwargs: Optional[Dict] = None,
        scheduler_config: Optional[Dict] = None,
    ) -> None:
        """Construct `StandardModel`."""
        # Base class constructor
        super().__init__()

        # Check(s)
        if isinstance(tasks, Task):
            tasks = [tasks]
        assert isinstance(tasks, (list, tuple))
        assert all(isinstance(task, Task) for task in tasks)
        assert isinstance(detector, Detector)
        assert isinstance(gnn, GNN)
        assert coarsening is None or isinstance(coarsening, Coarsening)

        # Member variable(s)
        self._detector = detector
        self._gnn = gnn
        self._tasks = ModuleList(tasks)
        self._coarsening = coarsening
        self._optimizer_class = optimizer_class
        self._optimizer_kwargs = optimizer_kwargs or dict()
        self._scheduler_class = scheduler_class
        self._scheduler_kwargs = scheduler_kwargs or dict()
        self._scheduler_config = scheduler_config or dict()
        self._dataset = dataset
        self._max_epochs = max_epochs

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure the model's optimizer(s)."""
        optimizer = self._optimizer_class(
            self.parameters(), **self._optimizer_kwargs
        )
        config = {
            "optimizer": optimizer,
        }
        if self._scheduler_class is not None:
            scheduler = self._scheduler_class(
                optimizer, **self._scheduler_kwargs
            )
            config.update(
                {
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        **self._scheduler_config,
                    },
                }
            )
        return config

    def forward(self, data: Data) -> List[Union[Tensor, Data]]:
        """Forward pass, chaining model components."""
        #import pdb;pdb.set_trace()
        if self._coarsening:
            data = self._coarsening(data)
        data = self._detector(data)
        x = self._gnn(data)
        #preds = [task(x) for task in self._tasks]
        if USE_ALL_FEA_IN_PRED:
            preds = [task(x) for task in self._tasks] + [x]
        else:
            preds = [task(x) for task in self._tasks]
        #preds = [self._tasks[0](x)]
        return preds

    def training_step(self, train_batch: Data, batch_idx: int) -> Tensor:
        """Perform training step."""
        preds = self(train_batch)
        vlosses = self._tasks[1].compute_loss(preds[1], train_batch)
        vloss = torch.sum(vlosses)
        
        tlosses = self._tasks[0].compute_loss(preds[0], train_batch)
        tloss = torch.sum(tlosses)

        #x = self.current_epoch/self._max_epochs
        #x = 0.5 + x/2
        #y = 1-x
        #loss = vloss*y + tloss*x
        loss = vloss + tloss
        return {"loss": loss, 'vloss': vloss, 'tloss': tloss}

    def validation_step(self, val_batch: Data, batch_idx: int) -> Tensor:
        """Perform validation step."""
        preds = self(val_batch)
        vlosses = self._tasks[1].compute_loss(preds[1], val_batch)
        vloss = torch.sum(vlosses)
        
        tlosses = self._tasks[0].compute_loss(preds[0], val_batch)
        tloss = torch.sum(tlosses)
        loss = vloss + tloss
        return {"loss": loss, 'vloss': vloss, 'tloss': tloss}

    def _get_batch_size(self, data: Data) -> int:
        return torch.numel(torch.unique(data.batch))

    def inference(self) -> None:
        """Activate inference mode."""
        for task in self._tasks:
            task.inference()

    def train(self, mode: bool = True) -> "Model":
        """Deactivate inference mode."""
        super().train(mode)
        if mode:
            for task in self._tasks:
                task.train_eval()
        return self

    def predict(
        self,
        dataloader: DataLoader,
        gpus: Optional[Union[List[int], int]] = None,
        distribution_strategy: Optional[str] = None,
    ) -> List[Tensor]:
        """Return predictions for `dataloader`."""
        self.inference()
        return super().predict(
            dataloader=dataloader,
            gpus=gpus,
            distribution_strategy=distribution_strategy,
        )
    
    def training_epoch_end(self, training_step_outputs):
        loss = torch.stack([x["loss"] for x in training_step_outputs]).mean()
        vloss = torch.stack([x["vloss"] for x in training_step_outputs]).mean()
        tloss = torch.stack([x["tloss"] for x in training_step_outputs]).mean()
        self.log_dict(
            {"trn_loss": loss, "trn_vloss": vloss, "trn_tloss": tloss},
            prog_bar=True,
            sync_dist=True,
        )
        print(f'epoch:{self.current_epoch}, train loss:{loss.item()}, tloss:{tloss.item()}, vloss:{vloss.item()}')
        self._dataset.reset_epoch()
        
    def validation_epoch_end(self, validation_step_outputs):
        loss = torch.stack([x["loss"] for x in validation_step_outputs]).mean()
        vloss = torch.stack([x["vloss"] for x in validation_step_outputs]).mean()
        tloss = torch.stack([x["tloss"] for x in validation_step_outputs]).mean()
        self.log_dict(
            {"val_loss": loss, "val_vloss": vloss, "val_tloss": tloss},
            prog_bar=True,
            sync_dist=True,
        )
        print(f'epoch:{self.current_epoch}, valid loss:{loss.item()}, tloss:{tloss.item()}, vloss:{vloss.item()}')





class EdgeConv0(MessagePassing):
    r"""The edge convolutional operator from the `"Dynamic Graph CNN for
    Learning on Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)}
        h_{\mathbf{\Theta}}(\mathbf{x}_i \, \Vert \,
        \mathbf{x}_j - \mathbf{x}_i),

    where :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* a MLP.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps pair-wise concatenated node features :obj:`x` of shape
            :obj:`[-1, 2 * in_channels]` to shape :obj:`[-1, out_channels]`,
            *e.g.*, defined by :class:`torch.nn.Sequential`.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"max"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V}|, F_{in}), (|\mathcal{V}|, F_{in}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
    def __init__(self, nn: Callable, aggr: str = 'max', **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x, size=None)

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.nn(torch.cat([x_i, x_j - x_i, x_j], dim=-1)) ##edgeConv0

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'
    
NODE_EDGE_FEA_RATIO = 0.7

class EdgeConv1(MessagePassing):
    r"""The edge convolutional operator from the `"Dynamic Graph CNN for
    Learning on Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)}
        h_{\mathbf{\Theta}}(\mathbf{x}_i \, \Vert \,
        \mathbf{x}_j - \mathbf{x}_i),

    where :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* a MLP.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps pair-wise concatenated node features :obj:`x` of shape
            :obj:`[-1, 2 * in_channels]` to shape :obj:`[-1, out_channels]`,
            *e.g.*, defined by :class:`torch.nn.Sequential`.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"max"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V}|, F_{in}), (|\mathcal{V}|, F_{in}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
    def __init__(self, nn: Callable, aggr: str = 'max', **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x, size=None)

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        edge_cnt = round(x_i.shape[-1]*NODE_EDGE_FEA_RATIO)
        if edge_cnt < 20:
            edge_cnt = 4                   #TODO　通常edge素性の数は20以上、それより少ないときは最初のレイヤーなのでXYZTの4つ
        edge_ij = torch.cat([(x_j - x_i)[:,:edge_cnt],x_j[:,edge_cnt:]], axis=-1)
        return self.nn(torch.cat([x_i, edge_ij], dim=-1)) ##edgeConv1

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'


USE_TRANS_IN_DYN1=True

class dynTrans1(EdgeConv0, LightningModule):
    """Dynamical edge convolution layer."""

    def __init__(
        self,
        layer_sizes,
        aggr: str = "max",
        nb_neighbors: int = 8,
        features_subset: Optional[Union[Sequence[int], slice]] = None,
        **kwargs: Any,
    ):
        """Construct `DynEdgeConv`.

        Args:
            nn: The MLP/torch.Module to be used within the `EdgeConv`.
            aggr: Aggregation method to be used with `EdgeConv`.
            nb_neighbors: Number of neighbours to be clustered after the
                `EdgeConv` operation.
            features_subset: Subset of features in `Data.x` that should be used
                when dynamically performing the new graph clustering after the
                `EdgeConv` operation. Defaults to all features.
            **kwargs: Additional features to be passed to `EdgeConv`.
        """
        # Check(s)
        if features_subset is None:
            features_subset = slice(None)  # Use all features
        assert isinstance(features_subset, (list, slice))
                
        layers = []
        for ix, (nb_in, nb_out) in enumerate(
            zip(layer_sizes[:-1], layer_sizes[1:])
        ):
            if ix == 0:
                nb_in *= 3 # edgeConv1
            layers.append(torch.nn.Linear(nb_in, nb_out))
            layers.append(torch.nn.LeakyReLU())
        d_model = nb_out
        # Base class constructor
        super().__init__(nn=torch.nn.Sequential(*layers), aggr=aggr, **kwargs)

        # Additional member variables
        self.nb_neighbors = nb_neighbors
        self.features_subset = features_subset
        

        self.norm_first=False
        
        self.norm1 = LayerNorm(d_model, eps=1e-5) #lNorm
        
        # Transformer layer(s)
        if USE_TRANS_IN_DYN1:
            encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True, dropout=DROPOUT, norm_first=self.norm_first)
            self._transformer_encoder = TransformerEncoder(encoder_layer, num_layers=1)        
        
        

    def forward(
        self, x: Tensor, edge_index: Adj, batch: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass."""
        # Standard EdgeConv forward pass
        
        if self.norm_first:
            x = self.norm1(x) # lNorm
            
        x_out = super().forward(x, edge_index)
        
        if x_out.shape[-1] == x.shape[-1] and SERIAL_CONNECTION:
            x = x + x_out
        else:
            x = x_out
            
        if not self.norm_first:
            x = self.norm1(x) # lNorm

        # Recompute adjacency
        edge_index = None

        # Transformer layer
        if USE_TRANS_IN_DYN1:
            x, mask = to_dense_batch(x, batch)
            x = self._transformer_encoder(x, src_key_padding_mask=~mask)
            x = x[mask]

        return x, edge_index

GLOBAL_POOLINGS = {
    "min": scatter_min,
    "max": scatter_max,
    "sum": scatter_sum,
    "mean": scatter_mean,
}

class DynEdge(GNN):
    """DynEdge (dynamical edge convolutional) model."""

    @save_model_config
    def __init__(
        self,
        nb_inputs: int,
        *,
        nb_neighbours: int = 8,
        features_subset: Optional[Union[List[int], slice]] = None,
        dynedge_layer_sizes: Optional[List[Tuple[int, ...]]] = None,
        post_processing_layer_sizes: Optional[List[int]] = None,
        readout_layer_sizes: Optional[List[int]] = None,
        global_pooling_schemes: Optional[Union[str, List[str]]] = None,
        add_global_variables_after_pooling: bool = False,
    ):
        """Construct `DynEdge`.

        Args:
            nb_inputs: Number of input features on each node.
            nb_neighbours: Number of neighbours to used in the k-nearest
                neighbour clustering which is performed after each (dynamical)
                edge convolution.
            features_subset: The subset of latent features on each node that
                are used as metric dimensions when performing the k-nearest
                neighbours clustering. Defaults to [0,1,2].
            dynedge_layer_sizes: The layer sizes, or latent feature dimenions,
                used in the `DynEdgeConv` layer. Each entry in
                `dynedge_layer_sizes` corresponds to a single `DynEdgeConv`
                layer; the integers in the corresponding tuple corresponds to
                the layer sizes in the multi-layer perceptron (MLP) that is
                applied within each `DynEdgeConv` layer. That is, a list of
                size-two tuples means that all `DynEdgeConv` layers contain a
                two-layer MLP.
                Defaults to [(128, 256), (336, 256), (336, 256), (336, 256)].
            post_processing_layer_sizes: Hidden layer sizes in the MLP
                following the skip-concatenation of the outputs of each
                `DynEdgeConv` layer. Defaults to [336, 256].
            readout_layer_sizes: Hidden layer sizes in the MLP following the
                post-processing _and_ optional global pooling. As this is the
                last layer(s) in the model, the last layer in the read-out
                yields the output of the `DynEdge` model. Defaults to [128,].
            global_pooling_schemes: The list global pooling schemes to use.
                Options are: "min", "max", "mean", and "sum".
            add_global_variables_after_pooling: Whether to add global variables
                after global pooling. The alternative is to  added (distribute)
                them to the individual nodes before any convolutional
                operations.
        """
        # Latent feature subset for computing nearest neighbours in DynEdge.
        if features_subset is None:
            features_subset = slice(0, 4) #4D

        # DynEdge layer sizes
        if dynedge_layer_sizes is None: #nb_nearest_neighboursと合わせて変更
            dynedge_layer_sizes = DYNEDGE_LAYER_SIZE

        assert isinstance(dynedge_layer_sizes, list)
        assert len(dynedge_layer_sizes)
        assert all(isinstance(sizes, tuple) for sizes in dynedge_layer_sizes)
        assert all(len(sizes) > 0 for sizes in dynedge_layer_sizes)
        assert all(
            all(size > 0 for size in sizes) for sizes in dynedge_layer_sizes
        )

        self._dynedge_layer_sizes = dynedge_layer_sizes

        # Post-processing layer sizes
        if post_processing_layer_sizes is None:
            post_processing_layer_sizes = [
                336,
                256,
            ]

        assert isinstance(post_processing_layer_sizes, list)
        assert len(post_processing_layer_sizes)
        assert all(size > 0 for size in post_processing_layer_sizes)

        self._post_processing_layer_sizes = post_processing_layer_sizes

        # Read-out layer sizes
        if readout_layer_sizes is None:
            readout_layer_sizes = [
                256,
                128,
            ]

        assert isinstance(readout_layer_sizes, list)
        assert len(readout_layer_sizes)
        assert all(size > 0 for size in readout_layer_sizes)

        self._readout_layer_sizes = readout_layer_sizes
        


        # Global pooling scheme(s)
        if isinstance(global_pooling_schemes, str):
            global_pooling_schemes = [global_pooling_schemes]

        if isinstance(global_pooling_schemes, list):
            for pooling_scheme in global_pooling_schemes:
                assert (
                    pooling_scheme in GLOBAL_POOLINGS
                ), f"Global pooling scheme {pooling_scheme} not supported."
        else:
            assert global_pooling_schemes is None

        self._global_pooling_schemes = global_pooling_schemes

        if add_global_variables_after_pooling:
            assert self._global_pooling_schemes, (
                "No global pooling schemes were request, so cannot add global"
                " variables after pooling."
            )
        self._add_global_variables_after_pooling = (
            add_global_variables_after_pooling
        )

        # Base class constructor
        super().__init__(nb_inputs, self._readout_layer_sizes[-1])

        # Remaining member variables()
        self._activation = torch.nn.LeakyReLU()
        self._nb_inputs = nb_inputs
        self._nb_global_variables = 5 + nb_inputs
        self._nb_neighbours = nb_neighbours
        self._features_subset = features_subset

        self._construct_layers()

    def _construct_layers(self) -> None:
        """Construct layers (torch.nn.Modules)."""
        # Convolutional operations
        nb_input_features = self._nb_inputs
        if USE_G:
            if not self._add_global_variables_after_pooling:
                nb_input_features += self._nb_global_variables

        self._conv_layers = torch.nn.ModuleList()
        nb_latent_features = nb_input_features
        for sizes in self._dynedge_layer_sizes:
            conv_layer = dynTrans1(
                [nb_latent_features] + list(sizes),
                aggr="max",
                nb_neighbors=self._nb_neighbours,
                features_subset=self._features_subset,
            )
            self._conv_layers.append(conv_layer)
            nb_latent_features = sizes[-1]

        # Post-processing operations
        if SERIAL_CONNECTION:
            nb_latent_features = self._dynedge_layer_sizes[-1][-1]
        else:
            nb_latent_features = (
                sum(sizes[-1] for sizes in self._dynedge_layer_sizes)
                + nb_input_features
            )

        if USE_PP:
            post_processing_layers = []
            layer_sizes = [nb_latent_features] + list(
                self._post_processing_layer_sizes
            )
            for nb_in, nb_out in zip(layer_sizes[:-1], layer_sizes[1:]):
                post_processing_layers.append(torch.nn.Linear(nb_in, nb_out))
                post_processing_layers.append(self._activation)
            last_posting_layer_output_dim = nb_out

            self._post_processing = torch.nn.Sequential(*post_processing_layers)
        else:
            last_posting_layer_output_dim = nb_latent_features

        # Read-out operations
        nb_poolings = (
            len(self._global_pooling_schemes)
            if self._global_pooling_schemes
            else 1
        )
        nb_latent_features = last_posting_layer_output_dim * nb_poolings
        if USE_G:
            if self._add_global_variables_after_pooling:  
                nb_latent_features += self._nb_global_variables

        readout_layers = []
        layer_sizes = [nb_latent_features] + list(self._readout_layer_sizes)
        for nb_in, nb_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            readout_layers.append(torch.nn.Linear(nb_in, nb_out))
            readout_layers.append(self._activation)

        self._readout = torch.nn.Sequential(*readout_layers)
        

        # Transformer layer(s)
        if USE_TRANS_IN_LAST:
            encoder_layer = TransformerEncoderLayer(d_model=last_posting_layer_output_dim, nhead=8, batch_first=True, dropout=DROPOUT, norm_first=False)
            self._transformer_encoder = TransformerEncoder(encoder_layer, num_layers=USE_TRANS_IN_LAST)        


    def _global_pooling(self, x: Tensor, batch: LongTensor) -> Tensor:
        """Perform global pooling."""
        assert self._global_pooling_schemes
        pooled = []
        for pooling_scheme in self._global_pooling_schemes:
            pooling_fn = GLOBAL_POOLINGS[pooling_scheme]
            pooled_x = pooling_fn(x, index=batch, dim=0)
            if isinstance(pooled_x, tuple) and len(pooled_x) == 2:
                # `scatter_{min,max}`, which return also an argument, vs.
                # `scatter_{mean,sum}`
                pooled_x, _ = pooled_x
            pooled.append(pooled_x)

        return torch.cat(pooled, dim=1)

    def _calculate_global_variables(
        self,
        x: Tensor,
        edge_index: LongTensor,
        batch: LongTensor,
        *additional_attributes: Tensor,
    ) -> Tensor:
        """Calculate global variables."""
        # Calculate homophily (scalar variables)
        h_x, h_y, h_z, h_t = calculate_xyzt_homophily(x, edge_index, batch)

        # Calculate mean features
        global_means = scatter_mean(x, batch, dim=0)

        # Add global variables
        global_variables = torch.cat(
            [
                global_means,
                h_x,
                h_y,
                h_z,
                h_t,
            ]
            + [attr.unsqueeze(dim=1) for attr in additional_attributes],
            dim=1,
        )

        return global_variables

    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass."""
        # Convenience variables
        x, edge_index, batch = data.x, data.edge_index, data.batch

        
        if USE_G:
            global_variables = self._calculate_global_variables(
                x,
                edge_index[0],
                batch,
                torch.log10(data.n_pulses),
            )

            # Distribute global variables out to each node
            if not self._add_global_variables_after_pooling:
                distribute = (
                    batch.unsqueeze(dim=1) == torch.unique(batch).unsqueeze(dim=0)
                ).type(torch.float)

                global_variables_distributed = torch.sum(
                    distribute.unsqueeze(dim=2)
                    * global_variables.unsqueeze(dim=0),
                    dim=1,
                )

                x = torch.cat((x, global_variables_distributed), dim=1)


        if SERIAL_CONNECTION:
            for conv_layer_index, conv_layer in enumerate(self._conv_layers):
                x, _edge_index = conv_layer(x, data.edge_index[0], batch)
        else:
            skip_connections = [x]
            for conv_layer_index, conv_layer in enumerate(self._conv_layers):
                x, _edge_index = conv_layer(x, data.edge_index[0], batch)
                #print('    dynEdge output skip_connections', x.shape)
                skip_connections.append(x)

            # Skip-cat
            x = torch.cat(skip_connections, dim=1)

        if USE_TRANS_IN_LAST:
            x, mask = to_dense_batch(x, batch)
            x = self._transformer_encoder(x, src_key_padding_mask=~mask)
            x = x[mask]
        
        if USE_PP:
            x = self._post_processing(x)
        
        if self._global_pooling_schemes:
            x = self._global_pooling(x, batch=batch)
            if USE_G:
                if self._add_global_variables_after_pooling:
                    x = torch.cat(
                        [
                            x,
                            global_variables,
                        ],
                        dim=1,
                    )

        x = self._readout(x)

        return x


class GraphBuilder(Model):  # pylint: disable=too-few-public-methods
    """Base class for graph building."""

    pass



class KNNGraphBuilderMulti(GraphBuilder):  # pylint: disable=too-few-public-methods
    """Builds graph from the k-nearest neighbours."""

    @save_model_config
    def __init__(
        self,
        nb_nearest_neighbours,
        columns,
    ):
        """Construct `KNNGraphBuilder`."""
        # Base class constructor
        super().__init__()

        # Member variable(s)
        assert len(nb_nearest_neighbours) == len(columns)
        self._nb_nearest_neighbours = nb_nearest_neighbours
        self._columns = columns

    def forward(self, data: Data) -> Data:
        """Forward pass."""
        # Constructs the adjacency matrix from the raw, DOM-level data and
        # returns this matrix
        
        if data.edge_index is not None:
            self.info(
                "WARNING: GraphBuilder received graph with pre-existing "
                "structure. Will overwrite."
            )
        edge_index_list = []
        x = data.x
        TIME_PARAM_FOR_DIST = 1/10
        x[:,3] = x[:,3]*TIME_PARAM_FOR_DIST
        for idx in range(len(self._nb_nearest_neighbours)):
            nb_nearest_neighbour = self._nb_nearest_neighbours[idx]
            if type(nb_nearest_neighbour) == str:
                nb_nearest_neighbour_min, nb_nearest_neighbour_max = nb_nearest_neighbour.split('-')
                nb_nearest_neighbour = torch.randint(int(nb_nearest_neighbour_min), int(nb_nearest_neighbour_max), (1,))[0]
            elif type(nb_nearest_neighbour) == list:
                nb_nearest_neighbour = random.choice(nb_nearest_neighbour)
            edge_index = knn_graph(
                x[:, self._columns[idx]]/1000,
                nb_nearest_neighbour,
                data.batch,
            ).to(self.device)
            edge_index_list.append(edge_index)
        x[:,3] = x[:,3]/TIME_PARAM_FOR_DIST

        data.edge_index = edge_index_list
        return data


# logger = Logger("GraphNet")

class DistanceLoss2(LossFunction):

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        target = target.reshape(-1, 3)
        # Check(s)
        assert prediction.dim() == 2 and prediction.size()[1] == 4
        assert target.dim() == 2
        assert prediction.size()[0] == target.size()[0]
        
        eps = 1e-4
        prediction_length = torch.linalg.vector_norm(prediction[:, [0, 1, 2]], dim=1)
        prediction_length = torch.clamp(prediction_length, min=eps)
        prediction =  prediction[:, [0, 1, 2]]/prediction_length.unsqueeze(1)
        cosLoss = prediction[:, 0] * target[:, 0] + prediction[:, 1] * target[:, 1] + prediction[:, 2] * target[:, 2]    
        cosLoss = torch.clamp(cosLoss, min=-1+eps, max=1-eps)
        thetaLoss = torch.arccos(cosLoss)
        return thetaLoss
    
    

class IceCubeKaggle2(Detector):
    """`Detector` class for Kaggle Competition."""

    # Implementing abstract class attribute
    features = FEATURES.KAGGLE

    def _forward(self, data: Data) -> Data:
        """Ingest data, build graph, and preprocess features.

        Args:
            data: Input graph data.

        Returns:
            Connected and preprocessed graph data.
        """
        # Check(s)
        self._validate_features(data)

        # Preprocessing
        data.x[:, 0] /= 500.0  # x
        data.x[:, 1] /= 500.0  # y
        data.x[:, 2] /= 500.0  # z
        data.x[:, 3] = (data.x[:, 3] - 1.0e04) / (500.0*0.23)  # 0.23 is speed of light in ice
        data.x[:, 4] = torch.log10(data.x[:, 4]) / 3.0  # charge

        return data
    
class DirectionReconstructionWithKappa2(Task):
    """Reconstructs direction with kappa from the 3D-vMF distribution."""

    # Requires three features: untransformed points in (x,y,z)-space.
    nb_inputs = 3

    def _forward(self, x: Tensor) -> Tensor:
        # Transform outputs to angle and prepare prediction
        kappa = torch.linalg.vector_norm(x, dim=1)# + eps_like(x)
        kappa = torch.clamp(kappa, min=torch.finfo(x.dtype).eps)
        vec_x = x[:, 0] / kappa
        vec_y = x[:, 1] / kappa
        vec_z = x[:, 2] / kappa
        return torch.stack((vec_x, vec_y, vec_z, kappa), dim=1)

def build_model2(config: Dict[str,Any], train_dataloader: Any, train_dataset: Any) -> StandardModel2:
    """Builds GNN from config"""
    # Building model
    detector = IceCubeKaggle2(
        graph_builder=KNNGraphBuilderMulti(nb_nearest_neighbours=NB_NEAREST_NEIGHBOURS, columns=COLUMNS_NEAREST_NEIGHBOURS) #dynedge_layer_sizes と合わせて変更
    )
    gnn = DynEdge(
        nb_inputs=detector.nb_outputs,
        #global_pooling_schemes=["min", "max", "mean"],
        global_pooling_schemes=["max"],
        add_global_variables_after_pooling=True
    )

    if config["target"] == 'direction':
        task = DirectionReconstructionWithKappa2(
            hidden_size=gnn.nb_outputs,
            target_labels=config["target"],
            loss_function=VonMisesFisher3DLoss(),
        )
        task2 = DirectionReconstructionWithKappa2(
            hidden_size=gnn.nb_outputs,
            target_labels=config["target"],
            loss_function=DistanceLoss2(),
        )
        
        prediction_columns = [config["target"] + "_x", 
                              config["target"] + "_y", 
                              config["target"] + "_z", 
                              config["target"] + "_kappa" ]
        additional_attributes = ['zenith', 'azimuth', 'event_id']

    model = StandardModel2(
        detector=detector,
        gnn=gnn,
        tasks=[task2, task],
        dataset=train_dataset,
        max_epochs=config["fit"]["max_epochs"],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-03, "eps": 1e-03},
        #optimizer_class=Lion,
        #optimizer_kwargs={"lr": 1e-04},
        scheduler_class=PiecewiseLinearLR,
        scheduler_kwargs={
            "milestones": [
                0,
                10  * len(train_dataloader)//(len(config['gpus'])*config['accumulate_grad_batches'][0]),
                len(train_dataloader)*config["fit"]["max_epochs"]//(len(config['gpus'])*config['accumulate_grad_batches'][0]*2),
                len(train_dataloader)*config["fit"]["max_epochs"]//(len(config['gpus'])*config['accumulate_grad_batches'][0]),                
            ],
            "factors": [1e-03, 1, 1, 1e-03],
            "verbose": config["scheduler_verbose"],
        },
        scheduler_config={
            "interval": "step",
        },
    )
    model.prediction_columns = prediction_columns
    model.additional_attributes = additional_attributes
    
    return model

def load_pretrained_model(config: Dict[str,Any], state_dict_path: str = '/home/tito/kaggle/icecube-neutrinos-in-deep-ice/input/dynedge-pretrained/dynedge_pretrained_batch_1_to_50/state_dict.pth') -> StandardModel2:
    train_dataloader, _, train_dataset, _ = make_dataloaders2(config = config)
    model = build_model2(config = config, 
                        train_dataloader = train_dataloader,
                        train_dataset = train_dataset,
                        )
    #model._inference_trainer = Trainer(config['fit'])
    model.load_state_dict(state_dict_path)
    model.prediction_columns = [config["target"] + "_x", 
                              config["target"] + "_y", 
                              config["target"] + "_z", 
                              config["target"] + "_kappa" ]
    model.additional_attributes = ['zenith', 'azimuth', 'event_id']
    return model

def make_dataloaders2(config: Dict[str, Any]) -> List[Any]:
    """Constructs training and validation dataloaders for training with early stopping."""
    train_dataloader, train_dataset = make_dataloader2(db = "dummy",
                                            selection = None,
                                            pulsemaps = config['pulsemap'],
                                            features = features,
                                            truth = truth,
                                            batch_ids = config['train_batch_ids'],
                                            batch_size = config['batch_size'],
                                            num_workers = config['num_workers'],
                                            shuffle = True,
                                            labels = {'direction': Direction()},
                                            index_column = config['index_column'],
                                            truth_table = config['truth_table'],
                                            max_len = config['train_len'],
                                            max_pulse = config['train_max_pulse'],
                                            min_pulse = config['train_min_pulse'],
                                            )
    
    validate_dataloader, validate_dataset = make_dataloader2(db = "dummy",
                                            selection = None,
                                            pulsemaps = config['pulsemap'],
                                            features = features,
                                            truth = truth,
                                            batch_ids = config['valid_batch_ids'],
                                            batch_size = config['batch_size'],
                                            num_workers = config['num_workers'],
                                            shuffle = False,
                                            labels = {'direction': Direction()},
                                            index_column = config['index_column'],
                                            truth_table = config['truth_table'],
                                            max_len = config['valid_len'],
                                            max_pulse = config['valid_max_pulse'],
                                            min_pulse = config['valid_min_pulse'],
                                          
                                            )
    return train_dataloader, validate_dataloader,  train_dataset, validate_dataset


def inference(model, config: Dict[str, Any]) -> pd.DataFrame:
    """Applies model to the database specified in config['inference_database_path'] and saves results to disk."""
    # Make Dataloader
    test_dataloader = make_dataloader(db = config['inference_database_path'],
                                            selection = None, # Entire database
                                            pulsemaps = config['pulsemap'],
                                            features = features,
                                            truth = truth,
                                            batch_size = config['batch_size'],
                                            num_workers = config['num_workers'],
                                            shuffle = False,
                                            labels = {'direction': Direction()},
                                            index_column = config['index_column'],
                                            truth_table = config['truth_table'],
                                            )
    
    # Get predictions
    results = model.predict_as_dataframe(
        gpus = config['gpus'],
        dataloader = test_dataloader,
        prediction_columns=model.prediction_columns,
        additional_attributes=model.additional_attributes,
    )
    return results

BATCH_DIR = 'input/icecube-neutrinos-in-deep-ice/train'
META_DIR = 'input/icecube-neutrinos-in-deep-ice/train'
FILTER_BY_KAPPA_THRE = 0.5

# training setting

TRAIN_MODE = False
hostName = socket.gethostname()
DROPOUT=0.0
NB_NEAREST_NEIGHBOURS = [6]
COLUMNS_NEAREST_NEIGHBOURS = [slice(0,4)]
USE_G = True
ONLY_AUX_FALSE = False
SERIAL_CONNECTION = True
USE_PP = True
USE_TRANS_IN_LAST=0
DYNEDGE_LAYER_SIZE = [
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
            ]

# Constants
features = FEATURES.KAGGLE
truth = TRUTH.KAGGLE


runName = 'base1-3l250p4n-batch400-650x30-infer' #TODO

    
# Configuration
FORCE_MAX_PULSE = 6000 # 強制的にこの件数以上のパルスは返さない

project = "650x20"
n_batch = 650
n_round = 10

batch_size = 400                # multi GPU
gpus = [0]                     # multi GPU
accumulate_grad_batches = {0: 5}

if len(gpus) > 1:
    if is_env_notebook():
        distribution_strategy = 'ddp_notebook'
    else:
        distribution_strategy = 'ddp'
else:
    distribution_strategy = None



config = {
        "path": 'dummy',
        "inference_database_path": '/home/tito/kaggle/icecube-neutrinos-in-deep-ice/notebook_graphnet/batch_660.db', #dummy
        "pulsemap": 'pulse_table', #dummy
        "truth_table": 'meta_table', #dummy
        "features": features,
        "truth": truth,
        "index_column": 'event_id',
        "run_name_tag": 'my_example',
        "batch_size": batch_size,
        "num_workers": 2, #todo
        "target": 'direction',
        "early_stopping_patience": n_batch,
        "gpus": gpus,
        "fit": {
                "max_epochs": n_batch*n_round,
                "gpus": gpus,
                "distribution_strategy": distribution_strategy,
                "check_val_every_n_epoch":10,
                "precision": 16,
                #"gradient_clip_val": 0.9,
                "reload_dataloaders_every_n_epochs": 1,
                },

        "accumulate_grad_batches": accumulate_grad_batches,
        'runName': runName,
        'project': project,
        'scheduler_verbose': False,
        'train_batch_ids': list(range(1,n_batch+1)),
        'valid_batch_ids': [660], # only suport one batch
        'test_selection': None,
        'base_dir': 'training',
        'train_len': 0,                    #not using anymore
        'valid_len': 0,                    #not using anymore
        'train_max_pulse': 300,
        'valid_max_pulse': 200,
        'train_min_pulse': 0,
        'valid_min_pulse': 0,
}


debug = False # bbb
if debug:
    runName = runName + '_debug'
    config["project"] = 'debug'
    #config["num_workers"] = 0
    config["batch_size"] = 2
    config["train_len"] = 2
    config["valid_len"] = 2


if TRAIN_MODE:

    train_dataloader, validate_dataloader, train_dataset, validate_dataset = make_dataloaders2(config = config)
    model = build_model2(config, train_dataloader, train_dataset)


    # Training model
    callbacks = [
        ModelCheckpoint(
            dirpath='../model_checkpoint_graphnet/',
            filename=runName+'-{epoch:02d}-{val_tloss:.6f}',
            monitor= 'val_tloss',
            save_top_k = 30,
            every_n_epochs = 10,
            save_weights_only=False,
        ),
        ProgressBar(),
    ]

    if 'accumulate_grad_batches' in config and len(config['accumulate_grad_batches']) > 0:
        callbacks.append(GradientAccumulationScheduler(scheduling=config['accumulate_grad_batches']))

    if debug == False:
        config["fit"]["logger"] = pl_loggers.WandbLogger(project=config["project"], name=runName)
        callbacks.append(LearningRateMonitor(logging_interval='epoch'))

    #config["fit"]["profiler"] = PyTorchProfiler( output_filename='profiler_results.txt', trace_every_n_steps=1)


    model.fit(
        train_dataloader,
        validate_dataloader,
        callbacks=callbacks,
        **config["fit"],
    )
    model.save_state_dict(f'../model_graphnet/{runName}-last.pth')

validateMode = False
USE_ALL_FEA_IN_PRED=True
INF_DEVICES = 0
result1_files = []

def is_env_notebook():
    """Determine wheather is the environment Jupyter Notebook"""
    if 'get_ipython' not in globals():
        # Python shell
        return False
    env_name = get_ipython().__class__.__name__
    if env_name == 'TerminalInteractiveShell':
        # IPython shell
        return False
    # Jupyter Notebook
    return True


def convert_to_3d(df: pd.DataFrame) -> pd.DataFrame:
    """Converts zenith and azimuth to 3D direction vectors"""
    df['true_x'] = np.cos(df['azimuth']) * np.sin(df['zenith'])
    df['true_y'] = np.sin(df['azimuth'])*np.sin(df['zenith'])
    df['true_z'] = np.cos(df['zenith'])
    return df

def calculate_angular_error(df : pd.DataFrame) -> pd.DataFrame:
    """Calcualtes the opening angle (angular error) between true and reconstructed direction vectors"""
    df['angular_error'] = np.arccos(df['true_x']*df['direction_x'] + df['true_y']*df['direction_y'] + df['true_z']*df['direction_z'])
    return df

def infer(min_pulse, max_pulse, batch_size, this_batch_id):
    if validateMode:
        labels = {'direction': Direction()}
    else:
        labels = None
    print('labels', labels)
    test_dataloader, test_dataset = make_dataloader2(db = "dummy",
                                                selection = None,
                                                pulsemaps = config['pulsemap'],
                                                features = features,
                                                truth = truth,
                                                batch_ids = [this_batch_id],
                                                batch_size = batch_size,
                                                num_workers = config['num_workers'],
                                                shuffle = False,
                                                labels = labels,
                                                index_column = config['index_column'],
                                                truth_table = config['truth_table'],
                                                max_len = 0,
                                                max_pulse = max_pulse,
                                                min_pulse = min_pulse,
                                                )

    if len(test_dataset) == 0:
        print('skip inference')
        return pd.DataFrame()
    
    model = build_model2(config, test_dataloader, test_dataset)

    state_dict =  torch.load(CKPT, torch.device('cpu'))
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict)

    USE_ORIG_PRED = True
    if USE_ORIG_PRED:
        event_ids = []
        zenith = []
        azimuth = []
        preds = []
        print('start predict')
        with torch.no_grad():
            model.eval()
            model.to(f'cuda:{INF_DEVICES}')
            for batch in tqdm(test_dataloader):
                pred = model(batch.to(f'cuda:{INF_DEVICES}'))
                #preds.append(pred[0])
                if USE_ALL_FEA_IN_PRED:
                    preds.append(torch.cat(pred, axis=-1))
                else:
                    preds.append(pred[0])
                event_ids.append(batch.event_id)
                if validateMode:
                    zenith.append(batch.zenith)
                    azimuth.append(batch.azimuth)
        preds = torch.cat(preds).to('cpu').detach().numpy()
        #results = pd.DataFrame(preds, columns=model.prediction_columns)
        if USE_ALL_FEA_IN_PRED:
            if preds.shape[1] == 128+8:
                columns = ['direction_x','direction_y','direction_z','direction_kappa1','direction_x1','direction_y1','direction_z1','direction_kappa'] + [f'idx{i}' for i in range(128)]
            else:
                columns = ['direction_x','direction_y','direction_z','direction_kappa'] + [f'idx{i}' for i in range(128)]
        else:
            columns=model.prediction_columns
        results = pd.DataFrame(preds, columns=columns)
        results['event_id'] = torch.cat(event_ids).to('cpu').detach().numpy()
        if validateMode:
            results['zenith'] = torch.cat(zenith).to('cpu').numpy()
            results['azimuth'] = torch.cat(azimuth).to('cpu').numpy()
            
        del zenith, azimuth, event_ids, preds
    else:
        results = model.predict_as_dataframe(
            gpus = config['gpus'],
            dataloader = test_dataloader,
            prediction_columns=model.prediction_columns,
            additional_attributes=model.additional_attributes,
        )
    gc.collect()
    if validateMode:
        results = convert_to_3d(results)
        results = calculate_angular_error(results)
        print('angular_error',results["angular_error"].mean())
    return results


if validateMode:
    BATCH_DIR = 'input/icecube-neutrinos-in-deep-ice/train'
    #meta = pd.read_parquet('input/icecubedatas/train_meta_651to660.parquet')
    #META_DIR = 'work/test_valid10'
    meta = pd.read_parquet('input/icecubedatas/train_meta_656to660.parquet')
    META_DIR = 'work/test_valid5'
    #meta = pd.read_parquet('input/icecubedatas/train_meta_660.parquet')
    #META_DIR = 'work/test_valid1'
else:
    BATCH_DIR = 'input/icecube-neutrinos-in-deep-ice/test'
    META_DIR = 'work/test'
    meta = pd.read_parquet('input/icecube-neutrinos-in-deep-ice/test_meta.parquet')
    
WORK_DIR = META_DIR
os.makedirs(META_DIR, exist_ok = True)
CSV_META = f'{META_DIR}/test_meta.csv'
if not os.path.exists(CSV_META):
    meta.to_csv(CSV_META, index=False)
batch_ids = meta.batch_id.unique()
for batch_id in batch_ids:
    out_file = f'{META_DIR}/meta_{batch_id}.parquet'
    if os.path.exists(out_file):
        continue
    meta[meta.batch_id == batch_id].reset_index(drop=False).to_parquet(out_file, index=False)
del meta
_ = gc.collect()

ONLY_AUX_FALSE = False
runName = '4l-ph3'
CKPT = 'input/icecubedatas/base1-4l-lr2-batch1000-splitModel2-650x14-retryFromStart2-last.pth'
COLUMNS_NEAREST_NEIGHBOURS = [slice(0,3)]
USE_G = True
USE_PP = True
DYNEDGE_LAYER_SIZE = [
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
            ]

results_all_batch = []
for this_batch_id in batch_ids: #TODO
    result0 = infer(0, 96, 400, this_batch_id); gc.collect()
    result1 = infer(96, 140, 200, this_batch_id); gc.collect()
    result2 = infer(140, 300, 100, this_batch_id); gc.collect()
    result3 = infer(300, 1000, 20, this_batch_id); gc.collect()
    result4 = infer(1000, 2000, 2, this_batch_id); gc.collect() # left all with FORCE_MAX_PULSE
    result5 = infer(2000, 0, 1, this_batch_id); gc.collect() # left all with FORCE_MAX_PULSE
    results = pd.concat([result0, result1, result2, result3, result4, result5]).sort_values('event_id')
    results_all_batch.append(results)

    if validateMode:
        print('angular_error',this_batch_id, results["angular_error"].mean())

    del result0, result1, result2, result3, result4, result5, results
    gc.collect()

results_all_batch = pd.concat(results_all_batch).sort_values('event_id').reset_index(drop=True)
results_all_batch['event_id'] = results_all_batch['event_id'].astype(int)
results_all_batch.to_csv(f'{WORK_DIR}/{runName}_{validateMode}.csv', index=False)
result1_files.append(f'{WORK_DIR}/{runName}_{validateMode}.csv')

if validateMode:
    print('angular_error',batch_ids, results_all_batch["angular_error"].mean())

del results_all_batch
gc.collect()

ONLY_AUX_FALSE = False
runName = '4l-retry1-e2t10-ph3'
CKPT = 'input/icecubedatas/base1-4l-splitModel2-650x2-theta10-retryFromStart3-last.pth'
COLUMNS_NEAREST_NEIGHBOURS = [slice(0,3)]
USE_G = True
USE_PP = True
DYNEDGE_LAYER_SIZE = [
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
            ]

results_all_batch = []
for this_batch_id in batch_ids: #TODO
    result0 = infer(0, 96, 400, this_batch_id); gc.collect()
    result1 = infer(96, 140, 200, this_batch_id); gc.collect()
    result2 = infer(140, 300, 100, this_batch_id); gc.collect()
    result3 = infer(300, 1000, 20, this_batch_id); gc.collect()
    result4 = infer(1000, 2000, 2, this_batch_id); gc.collect() # left all with FORCE_MAX_PULSE
    result5 = infer(2000, 0, 1, this_batch_id); gc.collect() # left all with FORCE_MAX_PULSE
    results = pd.concat([result0, result1, result2, result3, result4, result5]).sort_values('event_id')
    results_all_batch.append(results)

    if validateMode:
        print('angular_error',this_batch_id, results["angular_error"].mean())

    del result0, result1, result2, result3, result4, result5, results
    gc.collect()

results_all_batch = pd.concat(results_all_batch).sort_values('event_id').reset_index(drop=True)
results_all_batch['event_id'] = results_all_batch['event_id'].astype(int)
results_all_batch.to_csv(f'{WORK_DIR}/{runName}_{validateMode}.csv', index=False)
result1_files.append(f'{WORK_DIR}/{runName}_{validateMode}.csv')

if validateMode:
    print('angular_error',batch_ids, results_all_batch["angular_error"].mean())

del results_all_batch
gc.collect()

ONLY_AUX_FALSE = False
runName = '3lnoPP-ph3'
CKPT = 'input/icecubedatas/base1-3l300p500b-noPP-650x2-retryFromStart2-epoch1299-val_tloss0.999595.ckpt' # 0.9689275622367859, 0.9656946659088135
COLUMNS_NEAREST_NEIGHBOURS = [slice(0,3)]
USE_G = False
USE_PP = False
DYNEDGE_LAYER_SIZE = [
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
            ]

results_all_batch = []
for this_batch_id in batch_ids: #TODO
    result0 = infer(0, 96, 400, this_batch_id); gc.collect()
    result1 = infer(96, 140, 200, this_batch_id); gc.collect()
    result2 = infer(140, 300, 100, this_batch_id); gc.collect()
    result3 = infer(300, 1000, 20, this_batch_id); gc.collect()
    result4 = infer(1000, 2000, 2, this_batch_id); gc.collect() # left all with FORCE_MAX_PULSE
    result5 = infer(2000, 0, 1, this_batch_id); gc.collect() # left all with FORCE_MAX_PULSE
    results = pd.concat([result0, result1, result2, result3, result4, result5]).sort_values('event_id')
    results_all_batch.append(results)

    if validateMode:
        print('angular_error',this_batch_id, results["angular_error"].mean())

    del result0, result1, result2, result3, result4, result5, results
    gc.collect()

results_all_batch = pd.concat(results_all_batch).sort_values('event_id').reset_index(drop=True)
results_all_batch['event_id'] = results_all_batch['event_id'].astype(int)
results_all_batch.to_csv(f'{WORK_DIR}/{runName}_{validateMode}.csv', index=False)
result1_files.append(f'{WORK_DIR}/{runName}_{validateMode}.csv')

if validateMode:
    print('angular_error',batch_ids, results_all_batch["angular_error"].mean())

del results_all_batch
gc.collect()

ONLY_AUX_FALSE = False
runName = '3l4n-avg'
CKPT = 'input/icecubedatas/base1-3l250p4n-batch1000-650x8-retryFromStart1-avg.pth' # 0.9705609679222107
COLUMNS_NEAREST_NEIGHBOURS = [slice(0,4)]
USE_G = True
USE_PP = True
DYNEDGE_LAYER_SIZE = [
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
            ]

results_all_batch = []
for this_batch_id in batch_ids: #TODO
    result0 = infer(0, 96, 400, this_batch_id); gc.collect()
    result1 = infer(96, 140, 200, this_batch_id); gc.collect()
    result2 = infer(140, 300, 100, this_batch_id); gc.collect()
    result3 = infer(300, 1000, 20, this_batch_id); gc.collect()
    result4 = infer(1000, 2000, 2, this_batch_id); gc.collect() # left all with FORCE_MAX_PULSE
    result5 = infer(2000, 0, 1, this_batch_id); gc.collect() # left all with FORCE_MAX_PULSE
    results = pd.concat([result0, result1, result2, result3, result4, result5]).sort_values('event_id')
    results_all_batch.append(results)

    if validateMode:
        print('angular_error',this_batch_id, results["angular_error"].mean())

    del result0, result1, result2, result3, result4, result5, results
    gc.collect()

results_all_batch = pd.concat(results_all_batch).sort_values('event_id').reset_index(drop=True)
results_all_batch['event_id'] = results_all_batch['event_id'].astype(int)
results_all_batch.to_csv(f'{WORK_DIR}/{runName}_{validateMode}.csv', index=False)
result1_files.append(f'{WORK_DIR}/{runName}_{validateMode}.csv')

if validateMode:
    print('angular_error',batch_ids, results_all_batch["angular_error"].mean())

del results_all_batch
gc.collect()

ONLY_AUX_FALSE = False
runName = '4l-splitModel2-650x12-ph3'
CKPT = 'input/icecubedatas/base1-4l-splitModel2-650x12-retryFromStart4-retry1-last.pth' #0.9661051034927368
COLUMNS_NEAREST_NEIGHBOURS = [slice(0,3)]
USE_G = True
USE_PP = True
DYNEDGE_LAYER_SIZE = [
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
            ]

results_all_batch = []
for this_batch_id in batch_ids: #TODO
    result0 = infer(0, 96, 400, this_batch_id); gc.collect()
    result1 = infer(96, 140, 200, this_batch_id); gc.collect()
    result2 = infer(140, 300, 100, this_batch_id); gc.collect()
    result3 = infer(300, 1000, 20, this_batch_id); gc.collect()
    result4 = infer(1000, 2000, 2, this_batch_id); gc.collect() # left all with FORCE_MAX_PULSE
    result5 = infer(2000, 0, 1, this_batch_id); gc.collect() # left all with FORCE_MAX_PULSE
    results = pd.concat([result0, result1, result2, result3, result4, result5]).sort_values('event_id')
    results_all_batch.append(results)

    if validateMode:
        print('angular_error',this_batch_id, results["angular_error"].mean())

    del result0, result1, result2, result3, result4, result5, results
    gc.collect()

results_all_batch = pd.concat(results_all_batch).sort_values('event_id').reset_index(drop=True)
results_all_batch['event_id'] = results_all_batch['event_id'].astype(int)
results_all_batch.to_csv(f'{WORK_DIR}/{runName}_{validateMode}.csv', index=False)
result1_files.append(f'{WORK_DIR}/{runName}_{validateMode}.csv')

if validateMode:
    print('angular_error',batch_ids, results_all_batch["angular_error"].mean())

del results_all_batch
gc.collect()

ONLY_AUX_FALSE = False
runName = '4l4D500p-batch500-splitModel2-650x20-ph3'
CKPT = 'input/icecubedatas/base1-4l4D500p-batch500-splitModel2-650x20-retry4-last.pth' # 0.9676517248153687
COLUMNS_NEAREST_NEIGHBOURS = [slice(0,4)]
USE_G = True
USE_PP = True
DYNEDGE_LAYER_SIZE = [
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
            ]

results_all_batch = []
for this_batch_id in batch_ids: #TODO
    result0 = infer(0, 96, 400, this_batch_id); gc.collect()
    result1 = infer(96, 140, 200, this_batch_id); gc.collect()
    result2 = infer(140, 300, 100, this_batch_id); gc.collect()
    result3 = infer(300, 1000, 20, this_batch_id); gc.collect()
    result4 = infer(1000, 2000, 2, this_batch_id); gc.collect() # left all with FORCE_MAX_PULSE
    result5 = infer(2000, 0, 1, this_batch_id); gc.collect() # left all with FORCE_MAX_PULSE
    results = pd.concat([result0, result1, result2, result3, result4, result5]).sort_values('event_id')
    results_all_batch.append(results)

    if validateMode:
        print('angular_error',this_batch_id, results["angular_error"].mean())

    del result0, result1, result2, result3, result4, result5, results
    gc.collect()

results_all_batch = pd.concat(results_all_batch).sort_values('event_id').reset_index(drop=True)
results_all_batch['event_id'] = results_all_batch['event_id'].astype(int)
results_all_batch.to_csv(f'{WORK_DIR}/{runName}_{validateMode}.csv', index=False)
result1_files.append(f'{WORK_DIR}/{runName}_{validateMode}.csv')

if validateMode:
    print('angular_error',batch_ids, results_all_batch["angular_error"].mean())

del results_all_batch
gc.collect()

def zeaz2xyz(az, ze):
    z = np.cos(ze)
    rz = np.sin(ze)
    x = rz*np.cos(az)
    y = rz*np.sin(az)
    return x, y, z

def xyz2zeaz(x,y,z):
    r = np.sqrt(x**2+y**2+z**2)
    #print('R', r)
    x = x/r
    y = y/r
    z = z/r
    ze = np.arccos(z)
    rz = np.sin(ze)
    az = np.arccos(x/rz)
    az = np.where(y < 0, np.pi*2-az, az)
    az = np.nan_to_num(az,0)
    return az, ze

PRED_DIR = 'work/'
META_DIR = 'input/icecube-neutrinos-in-deep-ice/train'
class MyDatasetFile(Dataset):
    def __init__(self, 
                 runNames, 
                 batch_ids,
                 tgt_cols = ['direction_x','direction_y','direction_z','direction_kappa', 'direction_x1','direction_y1','direction_z1','direction_kappa1'],
                 use_mid_fea = True,
                ):
        self.batch_ids = batch_ids
        self.runNames = runNames
        self.tgt_cols = tgt_cols
        self.use_mid_fea = use_mid_fea
        self.this_batch_id = -1
        self.this_batch_idx = -1
        self.reset_epoch()
        
    def reset_epoch(self) -> None:
        self.this_batch_idx += 1
        if self.this_batch_idx >= len(self.batch_ids):
            self.this_batch_idx = 0
            
        if self.this_batch_id == self.batch_ids[self.this_batch_idx]:
            print('skip reset epoch ', self.this_batch_id, self.this_batch_idx)
            return
        else:
            self.this_batch_id = self.batch_ids[self.this_batch_idx]
            print('reset epoch to batch_id:', self.this_batch_id, self.this_batch_idx)

        meta = pd.read_parquet(f'{META_DIR}/meta_{self.this_batch_id}.parquet').reset_index(drop=True)
        meta['pulse_count'] = np.log1p(meta.last_pulse_index - meta.first_pulse_index + 1)
        self.meta = meta
        
        result_list = []
        for runName in self.runNames:
            df = pd.read_pickle(f'{WORK_DIR}/{runName}_{self.this_batch_id}.pkl')
            if 'direction_kappa' in df:
                df['direction_kappa'] = np.log1p(df['direction_kappa'])
            if 'direction_kappa1' in df:
                df['direction_kappa1'] = np.log1p(df['direction_kappa1'])
            columns = self.tgt_cols
            if self.use_mid_fea:
                columns = columns + [f'idx{i}' for i in range(128)]
            result_list.append(df[columns].reset_index(drop=True))
        self.X = pd.concat(result_list, axis=1).values.astype('float32')

        self.event_ids = meta['event_id'].values
        if 'azimuth' in meta:
            #self.Y = np.stack(zeaz2xyz(results[0]['azimuth'], results[0]['zenith'])).T.astype('float32')
            self.Y = np.stack(zeaz2xyz(meta['azimuth'], meta['zenith'])).T.astype('float32')
            self.with_gt = True
        else:
            self.with_gt = False

    def __len__(self):
        return self.X.shape[0]
    
    def n_columns(self):
        return self.X.shape[1]

    def __getitem__(self, index):
        x = self.X[index]        
        event_id = self.event_ids[index]
        if self.with_gt:
            y = self.Y[index]
        else:
            y = None
        return x, y, event_id


TGT_COLS = ['direction_x','direction_y','direction_z','direction_kappa', 'direction_x1','direction_y1','direction_z1','direction_kappa1']
class MyDataset(Dataset):
    def __init__(self, results, meta, mode='train'):
        meta = meta.reset_index(drop=True)
        self.mode = mode
        result_list = []
        for result in results:
            if len(TGT_COLS):
                columns = TGT_COLS
            elif 'direction_x1' in result:
                print('use 8 fea')
                columns = ['direction_x','direction_y','direction_z','direction_kappa', 'direction_x1','direction_y1','direction_z1','direction_kappa1']
            else:
                columns = ['direction_x','direction_y','direction_z','direction_kappa']
            if USE_MID_FEA:
                columns = columns + [f'idx{i}' for i in range(128)]
            result_list.append(result[columns].reset_index(drop=True))
        self.X = pd.concat(result_list, axis=1).values.astype('float32')
        self.event_ids = results[0]['event_id'].values
        if 'azimuth' in meta:
            #self.Y = np.stack(zeaz2xyz(results[0]['azimuth'], results[0]['zenith'])).T.astype('float32')
            self.Y = np.stack(zeaz2xyz(meta['azimuth'], meta['zenith'])).T.astype('float32')
            self.with_gt = True
        else:
            self.with_gt = False

    def __len__(self):
        return self.X.shape[0]
    
    def n_columns(self):
        return self.X.shape[1]

    def __getitem__(self, index):
        x = self.X[index]
        event_id = self.event_ids[index]
        if self.mode == 'test':
            return x,0,event_id
        
        if self.with_gt:
            y = self.Y[index]
        else:
            y = None
        return x, y, event_id

class StandardModel3(Model):
    @save_model_config
    def __init__(
        self,
        *,
        tasks: Union[Task, List[Task]],
        n_input_fea, 
        dataset,
        optimizer_class: type = Adam,
        optimizer_kwargs: Optional[Dict] = None,
        scheduler_class: Optional[type] = None,
        scheduler_kwargs: Optional[Dict] = None,
        scheduler_config: Optional[Dict] = None,
    ) -> None:
        """Construct `StandardModel`."""
        # Base class constructor
        super().__init__()

        # Check(s)
        if isinstance(tasks, Task):
            tasks = [tasks]
        assert isinstance(tasks, (list, tuple))
        assert all(isinstance(task, Task) for task in tasks)

        # Member variable(s)
        self._tasks = ModuleList(tasks)
        self._optimizer_class = optimizer_class
        self._optimizer_kwargs = optimizer_kwargs or dict()
        self._scheduler_class = scheduler_class
        self._scheduler_kwargs = scheduler_kwargs or dict()
        self._scheduler_config = scheduler_config or dict()
        self._n_input_fea = n_input_fea
        self._dataset = dataset
        
        mlp_layers = []
        layer_sizes = [n_input_fea, HIDDEN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE] # todo1
        for nb_in, nb_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            mlp_layers.append(torch.nn.Linear(nb_in, nb_out))
            mlp_layers.append(torch.nn.LeakyReLU())
            mlp_layers.append(torch.nn.Dropout(DROPOUT_PH2_MODEL))
        last_posting_layer_output_dim = nb_out

        self._mlp = torch.nn.Sequential(*mlp_layers)

            

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure the model's optimizer(s)."""
        optimizer = self._optimizer_class(
            self.parameters(), **self._optimizer_kwargs
        )
        config = {
            "optimizer": optimizer,
        }
        if self._scheduler_class is not None:
            scheduler = self._scheduler_class(
                optimizer, **self._scheduler_kwargs
            )
            config.update(
                {
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        **self._scheduler_config,
                    },
                }
            )
        return config

    def forward(self, x):
        x = self._mlp(x)
        x = [task(x) for task in self._tasks]
        return x

    def training_step(self, xye, idx) -> Tensor:
        """Perform training step."""
        x,y,event_ids = xye
        preds = self(x)
        batch = Data(x=x, direction=y)
        vlosses = self._tasks[1].compute_loss(preds[1], batch)
        vloss = torch.sum(vlosses)
        
        tlosses = self._tasks[0].compute_loss(preds[0], batch)
        tloss = torch.sum(tlosses)

        loss = vloss*0.1 + tloss
        return {"loss": loss, 'vloss': vloss, 'tloss': tloss}

    def validation_step(self, xye, idx) -> Tensor:
        """Perform validation step."""
        x,y,event_ids = xye
        preds = self(x)
        batch = Data(x=x, direction=y)
        vlosses = self._tasks[1].compute_loss(preds[1], batch)
        vloss = torch.sum(vlosses)
        
        tlosses = self._tasks[0].compute_loss(preds[0], batch)
        tloss = torch.sum(tlosses)
        loss = vloss*0.1 + tloss
        return {"loss": loss, 'vloss': vloss, 'tloss': tloss}

    def inference(self) -> None:
        """Activate inference mode."""
        for task in self._tasks:
            task.inference()

    def train(self, mode: bool = True) -> "Model":
        """Deactivate inference mode."""
        super().train(mode)
        if mode:
            for task in self._tasks:
                task.train_eval()
        return self

    def predict(
        self,
        dataloader: DataLoader,
        gpus: Optional[Union[List[int], int]] = None,
        distribution_strategy: Optional[str] = None,
    ) -> List[Tensor]:
        """Return predictions for `dataloader`."""
        self.inference()
        return super().predict(
            dataloader=dataloader,
            gpus=gpus,
            distribution_strategy=distribution_strategy,
        )
    
    def training_epoch_end(self, training_step_outputs):
        loss = torch.stack([x["loss"] for x in training_step_outputs]).mean()
        vloss = torch.stack([x["vloss"] for x in training_step_outputs]).mean()
        tloss = torch.stack([x["tloss"] for x in training_step_outputs]).mean()
        self.log_dict(
            {"trn_loss": loss, "trn_vloss": vloss, "trn_tloss": tloss},
            prog_bar=True,
            sync_dist=True,
        )
        print(f'epoch:{self.current_epoch}, train loss:{loss.item()}, tloss:{tloss.item()}, vloss:{vloss.item()}')
        self._dataset.reset_epoch()
        
    def validation_epoch_end(self, validation_step_outputs):
        loss = torch.stack([x["loss"] for x in validation_step_outputs]).mean()
        vloss = torch.stack([x["vloss"] for x in validation_step_outputs]).mean()
        tloss = torch.stack([x["tloss"] for x in validation_step_outputs]).mean()
        self.log_dict(
            {"val_loss": loss, "val_vloss": vloss, "val_tloss": tloss},
            prog_bar=True,
            sync_dist=True,
        )
        print(f'epoch:{self.current_epoch}, valid loss:{loss.item()}, tloss:{tloss.item()}, vloss:{vloss.item()}')

def build_model3(config, dataloader, dataset) -> StandardModel2:
    """Builds GNN from config"""
    # Building model

    if config["target"] == 'direction':
        task = DirectionReconstructionWithKappa2(
            hidden_size=HIDDEN_SIZE,
            target_labels=config["target"],
            loss_function=VonMisesFisher3DLoss(),
        )
        task2 = DirectionReconstructionWithKappa2(
            hidden_size=HIDDEN_SIZE,
            target_labels=config["target"],
            loss_function=DistanceLoss2(),
        )
        
        prediction_columns = [config["target"] + "_x", 
                              config["target"] + "_y", 
                              config["target"] + "_z", 
                              config["target"] + "_kappa" ]
        additional_attributes = ['zenith', 'azimuth', 'event_id']

    model = StandardModel3(
        tasks=[task2, task],
        n_input_fea=N_INPUT_FEA,
        dataset=dataset,
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-03, "eps": 1e-03},
        #optimizer_class=Lion,
        #optimizer_kwargs={"lr": 1e-04},
        scheduler_class=PiecewiseLinearLR,
        scheduler_kwargs={
            "milestones": [
                0,
                10  * len(dataloader)//(len(config['gpus'])*config['accumulate_grad_batches'][0]),
                len(dataloader)*config["fit"]["max_epochs"]//(len(config['gpus'])*config['accumulate_grad_batches'][0]*2),
                len(dataloader)*config["fit"]["max_epochs"]//(len(config['gpus'])*config['accumulate_grad_batches'][0]),                
            ],
            "factors": [1e-03, 1, 1, 1e-03],
            "verbose": config["scheduler_verbose"],
        },
        scheduler_config={
            "interval": "step",
        },
    )
    model.prediction_columns = prediction_columns
    model.additional_attributes = additional_attributes
    
    return model


def infer2(test_dataloader, model):
    USE_ORIG_PRED = True
    if USE_ORIG_PRED:
        event_ids = []
        zenith = []
        azimuth = []
        preds = []
        print('start predict')
        with torch.no_grad():
            model.eval()
            model.to(f'cuda:{INF_DEVICES}')
            for batch in tqdm(test_dataloader):
                pred = model(batch[0].to(f'cuda:{INF_DEVICES}'))
                #preds.append(pred[0])
                if USE_ALL_FEA_IN_PRED:
                    preds.append(torch.cat(pred, axis=-1))
                else:
                    preds.append(pred[0])
                event_ids.append(batch[2])
                if validateMode:
                    azimuth.append(batch[1][0])
                    zenith.append(batch[1][1])
        preds = torch.cat(preds).to('cpu').detach().numpy()
        #results = pd.DataFrame(preds, columns=model.prediction_columns)
        if USE_ALL_FEA_IN_PRED:
            if preds.shape[1] == 128+8:
                columns = ['direction_x','direction_y','direction_z','direction_kappa1','direction_x1','direction_y1','direction_z1','direction_kappa'] + [f'idx{i}' for i in range(128)]
            else:
                columns = ['direction_x','direction_y','direction_z','direction_kappa'] + [f'idx{i}' for i in range(128)]
        else:
            if preds.shape[1] == 8:
                columns=['direction_x','direction_y','direction_z','direction_kappa1','direction_x1','direction_y1','direction_z1','direction_kappa']
            else:
                columns=['direction_x','direction_y','direction_z','direction_kappa']
        results = pd.DataFrame(preds, columns=columns)
        results['event_id'] = torch.cat(event_ids).to('cpu').detach().numpy()
            
        del zenith, azimuth, event_ids, preds
    else:
        results = model.predict_as_dataframe(
            gpus = config['gpus'],
            dataloader = test_dataloader,
            prediction_columns=model.prediction_columns,
            additional_attributes=model.additional_attributes,
        )
    gc.collect()
    return results


def calculate_angular_error2(df_pred, df_gt):
    df_gt = df_gt.reset_index(drop=True)
    df_pred = df_pred.reset_index(drop=True)
    df_pred['angular_error'] = np.arccos(df_gt['true_x']*df_pred['direction_x'] + df_gt['true_y']*df_pred['direction_y'] + df_gt['true_z']*df_pred['direction_z'])
    return df_pred

def convert_to_3d(df: pd.DataFrame) -> pd.DataFrame:
    """Converts zenith and azimuth to 3D direction vectors"""
    df['true_x'] = np.cos(df['azimuth']) * np.sin(df['zenith'])
    df['true_y'] = np.sin(df['azimuth'])*np.sin(df['zenith'])
    df['true_z'] = np.cos(df['zenith'])
    return df

import time

HIDDEN_SIZE = 512
DROPOUT_PH2_MODEL = 0.0
N_INPUT_FEA = 136*len(result1_files)

USE_ALL_FEA_IN_PRED=False
INF_DEVICES = 0
USE_MID_FEA = True
CKPT = 'input/icecubedatas/base3-ens20-last.pth' # 0.9640
CKPT = 'input/icecubedatas/base4-ens23-last.pth' # 0.9637
CKPT = 'input/icecubedatas/base5-ens26-last.pth' # 0.9635
CKPT = 'input/icecubedatas/base5-ens32-6model-3layer-659batch-last.pth' # 0.963? todo1 

chunksize=50000
meta_chunks = pd.read_csv(CSV_META, chunksize=chunksize)
chunks = [pd.read_csv(f, chunksize=chunksize) for f in result1_files]
result_list = []

first_flg = True
while True:
    start_time = time.time()
    try:
        result_df = []
        for c in chunks:
            df = next(c)
            if 'direction_kappa' in df:
                df['direction_kappa'] = np.log1p(df['direction_kappa'])
            if 'direction_kappa1' in df:
                df['direction_kappa1'] = np.log1p(df['direction_kappa1'])
            result_df.append(df)
        this_meta = next(meta_chunks)
        this_meta['pulse_count'] = np.log1p(this_meta.last_pulse_index - this_meta.first_pulse_index + 1)
        test_dataset = MyDataset(result_df, this_meta, mode='test')
        test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2)
        if first_flg:
            model = build_model3(config = config, dataloader = test_dataloader, dataset = test_dataset)
            state_dict =  torch.load(CKPT, torch.device('cpu'))
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            model.load_state_dict(state_dict)
            first_flg = False

        results2 = infer2(test_dataloader, model)
        if validateMode:
            this_meta = convert_to_3d(this_meta)
            results2 = calculate_angular_error2(results2, this_meta)
            print(results2.angular_error.mean())
        result_list.append(results2)
    except StopIteration:
        break
    del result_df
    gc.collect()
    print('total time', time.time()-start_time)
    
results2 = pd.concat(result_list)
if validateMode:
    print('all', results2.angular_error.mean())
    print('660', results2.angular_error[-153924:].mean())


def prepare_dataframe(df, angle_post_fix = '', vec_post_fix = '') -> pd.DataFrame:
    r = np.sqrt(df['direction_x'+ vec_post_fix]**2 + df['direction_y'+ vec_post_fix]**2 + df['direction_z' + vec_post_fix]**2)
    df['zenith' + angle_post_fix] = np.arccos(df['direction_z'+ vec_post_fix]/r)
    df['azimuth'+ angle_post_fix] = np.arctan2(df['direction_y'+ vec_post_fix],df['direction_x' + vec_post_fix]) #np.sign(results['true_y'])*np.arccos((results['true_x'])/(np.sqrt(results['true_x']**2 + results['true_y']**2)))
    df['azimuth'+ angle_post_fix][df['azimuth'  + angle_post_fix]<0] = df['azimuth'  + angle_post_fix][df['azimuth'  +  angle_post_fix]<0] + 2*np.pi 
    return df

results2 = prepare_dataframe(results2)
results2[['event_id','azimuth','zenith']].sort_values('event_id').to_csv("submission.csv", index = False)
results2[['event_id','azimuth','zenith']].sort_values('event_id')

