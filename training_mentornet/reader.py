from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import pickle
import numpy as np
import torch.utils.data as data_utils

class Dataset(data_utils.Dataset):
    def __init__(self, indir, split_name) -> None:
        super().__init__()
        self._data = pickle.load(open(os.path.join(indir, split_name + '.p'), 'rb'))
        self._num_examples = self._data.shape[0]
        self.feat_dim = self._data.shape[1] - 1
        self._epochs_completed = 0
        self._index_in_epoch = 0    

    @property
    def __len__(self) -> int:
        return self._num_examples
    
    @property
    def is_binary_label(self):
        unique_labels = np.unique(self._data[:, -1])
        if len(unique_labels) == 2 and (0 in unique_labels) and (
            1 in unique_labels):
            return True
        return False

    def __getitem__(self, index: int):
        return self._data[index]

def get_train_dataloader(data_path: str, split_name: str, device: str='cpu', batch_size: int=32, worker_num: int=2):
    return data_utils.DataLoader(Dataset(data_path, split_name), batch_size=batch_size, shuffle=True, num_workers=worker_num,pin_memory=True)
