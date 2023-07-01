import os.path
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import lmdb
import zstandard
from tqdm import tqdm
from matplotlib import pyplot as plt

from config import Paths, Config


class PepeDataset(Dataset):
    """ Dogs with watermarks dataset. """

    def __init__(self, dataset_name: str, paths: Paths, augments=None):
        self.path = paths.parsed_datasets + dataset_name
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
        condition = torch.tensor(condition.astype('float32'))

        return image, condition


if __name__ == '__main__':
    cfg = Config()
    dataset = PepeDataset(dataset_name='celeba', paths=Paths())
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=8)
    for batch in tqdm(dataloader, desc="Testing dataset... "):
        assert batch[0].shape[1:] == (3, cfg.image_size, cfg.image_size), batch[0].shape
        assert batch[1].shape[1:] == (40,), batch[1].shape
