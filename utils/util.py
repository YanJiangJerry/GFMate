import pdb
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import Dataset, Data
from torch_geometric.transforms import SVDFeatureReduction


def x_padding(data, out_dim):
    assert data.x.size(-1) <= out_dim
    incremental_dimension = out_dim - data.x.size(-1)
    zero_features = torch.zeros(
        (data.x.size(0), incremental_dimension),
        dtype=data.x.dtype,
        device=data.x.device,
    )
    data.x = torch.cat([data.x, zero_features], dim=-1)
    return data


class CustomDataset(Dataset):
    def __init__(self, data_list):
        # Initialize the dataset with the provided list of Data objects
        super().__init__()
        self.data_list = data_list
        self._indices = torch.arange(len(data_list))  # Initialize the indices attribute

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

    def indices(self):
        return self._indices


def x_padding_graph(dataset, out_dim):
    new_data_list = []

    for data in dataset:
        assert data.x.size(-1) <= out_dim
        incremental_dimension = out_dim - data.x.size(-1)
        zero_features = torch.zeros(
            (data.x.size(0), incremental_dimension),
            dtype=data.x.dtype,
            device=data.x.device,
        )
        data.x = torch.cat([data.x, zero_features], dim=-1)
        new_data_list.append(data)

    new_dataset = CustomDataset(new_data_list)
    return new_dataset


# def x_svd(data, out_dim, batch_size=1000):
#     assert data.x.size(-1) >= out_dim
#     reduction = SVDFeatureReduction(out_dim)

#     num_batches = (data.x.size(0) + batch_size - 1) // batch_size
#     reduced_data = []

#     for i in range(num_batches):
#         batch_data = data.x[i*batch_size:(i+1)*batch_size]
#         temp_data = Data(x=batch_data, edge_index=data.edge_index)
#         reduced_batch = reduction(temp_data)
#         reduced_data.append(reduced_batch.x)

#     data.x = torch.cat(reduced_data, dim=0)
#     return data


def x_svd(data, out_dim, batch_size=10000):
    assert (
        data.x.size(-1) >= out_dim
    ), f"Input dimension {data.x.size(-1)} is smaller than required {out_dim}"
    reduction = SVDFeatureReduction(out_dim)

    num_batches = (data.x.size(0) + batch_size - 1) // batch_size
    reduced_data = []

    for i in range(num_batches):
        batch_data = data.x[i * batch_size : (i + 1) * batch_size]
        temp_data = Data(x=batch_data, edge_index=data.edge_index)
        reduced_batch = reduction(temp_data)

        if reduced_batch.x.shape[1] < out_dim:
            pad_size = out_dim - reduced_batch.x.shape[1]
            reduced_batch.x = F.pad(reduced_batch.x, (0, pad_size), "constant", 0)
        elif reduced_batch.x.shape[1] > out_dim:
            reduced_batch.x = reduced_batch.x[:, :out_dim]

        reduced_data.append(reduced_batch.x)

    assert all(
        tensor.shape[1] == out_dim for tensor in reduced_data
    ), "Final output dimensions are inconsistent"

    data.x = torch.cat(reduced_data, dim=0)
    return data
