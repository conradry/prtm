from pathlib import Path
from typing import Iterator, List, Mapping, Union

import numpy as np
import pyarrow.parquet as pq
import torch
from proteome.models.design.proteinsolver.utils.protein_sequence import \
    seq_to_tensor
from torch_geometric.data import Data


def normalize_cart_distances(cart_distances):
    return (cart_distances - 6) / 12


def normalize_seq_distances(seq_distances):
    return (seq_distances - 0) / 68.1319


def transform_edge_attr(data):
    cart_distances = data.edge_attr
    cart_distances = normalize_cart_distances(cart_distances)
    seq_distances = (
        (data.edge_index[1] - data.edge_index[0]).to(torch.float).unsqueeze(1)
    )
    seq_distances = normalize_seq_distances(seq_distances)
    data.edge_attr = torch.cat([cart_distances, seq_distances], dim=1)
    return data


def iter_parquet_file(
    filename: Union[str, Path],
    extra_columns: List[str],
    extra_column_renames: Mapping[str, str],
) -> Iterator:
    columns = (
        ["sequence", "residue_idx_1_corrected", "residue_idx_2_corrected", "distances"]
        if not extra_columns
        else extra_columns
    )

    column_renames = (
        {"residue_idx_1_corrected": "row_index", "residue_idx_2_corrected": "col_index"}
        if not extra_column_renames
        else extra_column_renames
    )

    parquet_file_obj = pq.ParquetFile(filename)
    for row_group_idx in range(parquet_file_obj.num_row_groups):
        df = parquet_file_obj.read_row_group(row_group_idx, columns=columns).to_pandas()
        df = df.rename(columns=column_renames)
        for tup in df.itertuples():
            yield tup


def row_to_data(tup, add_reversed_edges=True) -> Data:
    seq = torch.tensor(
        seq_to_tensor(tup.sequence.replace("-", "").encode("ascii")), dtype=torch.long
    )
    if (seq == 20).sum() > 0:
        return None

    row_index = _to_torch(tup.row_index).to(torch.long)
    col_index = _to_torch(tup.col_index).to(torch.long)
    edge_attr = _to_torch(tup.distances).to(torch.float).unsqueeze(dim=1)

    # Remove self loops
    mask = row_index == col_index
    if mask.any():
        row_index = row_index[~mask]
        col_index = col_index[~mask]
        edge_attr = edge_attr[~mask, :]

    if add_reversed_edges:
        edge_index = torch.stack(
            [torch.cat([row_index, col_index]), torch.cat([col_index, row_index])],
            dim=0,
        )
        edge_attr = torch.cat([edge_attr, edge_attr])
    else:
        edge_index = torch.stack([row_index, col_index], dim=0)

    edge_index, edge_attr = remove_nans(edge_index, edge_attr)
    data = Data(x=seq, edge_index=edge_index, edge_attr=edge_attr)
    data = data.coalesce()

    assert not data.contains_self_loops()
    assert data.is_coalesced()
    assert data.is_undirected()

    for c in tup._fields:
        if c not in ["sequence", "row_index", "col_index", "distances"]:
            setattr(data, c, torch.tensor([getattr(tup, c)]))

    return data


def remove_nans(edge_index, edge_attr):
    na_mask = torch.isnan(edge_index).any(dim=0).squeeze()
    if na_mask.any():
        edge_index = edge_index[:, ~na_mask]
        edge_attr = edge_attr[~na_mask]
    return edge_index, edge_attr


def _to_torch(data_array):
    if isinstance(data_array, torch.Tensor):
        return data_array
    elif isinstance(data_array, np.ndarray):
        return torch.from_numpy(data_array)
    else:
        return torch.tensor(data_array)
