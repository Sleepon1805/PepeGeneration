import os
import lmdb
import torch
import pickle
import zstandard
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List

from config import Paths, DataConfig
from data.condition_utils import decode_condition


class PepeDataset(Dataset):
    def __init__(self, data_config: DataConfig, paths: Paths, augments=None):
        self.dataset_name = data_config.dataset_name
        self.path = paths.parsed_datasets + self.dataset_name + str(data_config.image_size)
        self.augmentations = augments

        self._init_database()
        # set None to make them pickleable -> allow num_workers > 1
        self.db = None
        self.decompressor = None

    def _init_database(self):
        assert os.path.exists(self.path)

        self.db = lmdb.open(self.path, subdir=True, readonly=True, lock=False, readahead=False,
                            meminit=False)
        self.decompressor = zstandard.ZstdDecompressor()

        # load metadata
        with self.db.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b'__len__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # load databases
        if self.db is None:
            self._init_database()

        # get items
        with self.db.begin(write=False) as txn:
            image, condition = pickle.loads(self.decompressor.decompress(txn.get(self.keys[index])))

        # augmentations
        if self.augmentations:
            image = self.augmentations(image)

        # to tensor
        image = transforms.ToTensor()(image)
        image = 2 * image - 1
        condition = torch.from_numpy(condition.astype('float32'))

        return image, condition

    def show_item(self, item, axis: plt.Axes = None):
        sample, condition = item

        image = sample.numpy().transpose((1, 2, 0))
        image = (image + 1) / 2

        features = decode_condition(self.dataset_name, condition)

        if axis:
            axis.imshow(image)
            axis.set_title(str(features))
            axis.axis('off')
            return axis
        else:
            plt.imshow(image)
            plt.title(str(features))
            plt.show()


if __name__ == '__main__':
    data_cfg = DataConfig(
        dataset_name='pepe',
        image_size=128,
    )
    dataset = PepeDataset(data_cfg, paths=Paths())

    one_item = dataset[137]
    dataset.show_item(one_item)

    dataloader = DataLoader(dataset, batch_size=16, num_workers=8)
    for batch in tqdm(dataloader, desc="Testing dataset... "):
        assert batch[0].shape[1:] == (3, data_cfg.image_size, data_cfg.image_size), batch[0].shape
        assert batch[1].shape[1:] == (40,), batch[1].shape
