import os
import lmdb
import torch
import pickle
import zstandard
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from config import Paths, Config


class PepeDataset(Dataset):
    def __init__(self, dataset_name: str, image_size: int, paths: Paths, augments=None):
        self.path = paths.parsed_datasets + dataset_name + str(image_size)
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

    @staticmethod
    def show_item(item):
        sample, condition = item

        image = sample.numpy().transpose((1, 2, 0))
        image = (image + 1) / 2

        plt.imshow(image)
        plt.title(condition)
        plt.show()
        print(condition)


if __name__ == '__main__':
    cfg = Config()
    dataset = PepeDataset(dataset_name='celeba', image_size=cfg.image_size, paths=Paths())

    one_item = dataset[137]
    dataset.show_item(one_item)

    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=8)
    for batch in tqdm(dataloader, desc="Testing dataset... "):
        assert batch[0].shape[1:] == (3, cfg.image_size, cfg.image_size), batch[0].shape
        assert batch[1].shape[1:] == (40,), batch[1].shape
