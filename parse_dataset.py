import os
import lmdb
import pickle
import cv2
import zstandard
from tqdm import tqdm
from matplotlib import pyplot as plt

from config import cfg


class DataParser:
    def __init__(self, config, dataset_name: str):
        self.source_data_path, self.save_path = self._init_data_paths(config, dataset_name)
        self.db = LMDBCreator(self.save_path)  # lmdb

    @staticmethod
    def _init_data_paths(config, dataset_name):
        # source data
        if dataset_name == 'pepe':
            source_data_path = config.pepe_data_path
        elif dataset_name == 'celeba':
            source_data_path = config.celeba_data_path
        else:
            raise ValueError(f"dataset_name must be 'pepe' or 'celeba', got {dataset_name}")
        assert os.path.exists(source_data_path), \
            f'Given path for source images {source_data_path} does not exist! Change it in config.py.'

        # parsed data
        save_path = config.parsed_datasets + dataset_name
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        return source_data_path, save_path

    def parse_and_save_dataset(self):
        # init dataset
        for sample_num, filename in enumerate(tqdm(os.listdir(self.source_data_path), desc='Parsing images...')):
            # read image as ndarray
            image = cv2.imread(self.source_data_path + filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, dsize=cfg.image_size, interpolation=cv2.INTER_CUBIC)

            # save image
            self.db.write_lmdb_sample(sample_num, image)

        # save length and keys
        self.db.write_lmdb_metadata(len(os.listdir(self.source_data_path)))


class LMDBCreator:
    def __init__(self, path, max_size=10e9):
        self.path = path
        self.max_size = int(max_size)

        self.db = self.init_lmdb()
        self.compressor = zstandard.ZstdCompressor()

    def init_lmdb(self):
        os.makedirs(self.path, exist_ok=True)
        db = lmdb.open(self.path, subdir=True, map_size=self.max_size, readonly=False, meminit=False, map_async=True)
        return db

    def write_lmdb_sample(self, index, item):
        txn = self.db.begin(write=True)
        item = self.compressor.compress(pickle.dumps(item, protocol=5))
        txn.put(u'{}'.format(index).encode('ascii'), item)
        txn.commit()
        self.db.sync()

    def write_lmdb_metadata(self, num_samples):
        keys = [u'{}'.format(index).encode('ascii') for index in range(num_samples)]
        with self.db.begin(write=True) as txn:
            txn.put(b'__keys__', pickle.dumps(keys, protocol=5))
            txn.put(b'__len__', pickle.dumps(num_samples, protocol=5))
        self.db.sync()
        self.db.close()


if __name__ == '__main__':
    dataset_names = [
        'pepe',
        'celeba'
    ]
    for dataset in dataset_names:
        dataparser = DataParser(config=cfg, dataset_name=dataset)
        dataparser.parse_and_save_dataset()

