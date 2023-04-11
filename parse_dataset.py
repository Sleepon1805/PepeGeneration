import os
import lmdb
import pickle
import cv2
import zstandard
from tqdm import tqdm
from matplotlib import pyplot as plt

from config import cfg


class DataParser:
    def __init__(self, config):
        assert os.path.exists(config.source_data), \
            f'Given path for original images {config.source_data} does not exist! Change it in config.py.'
        if not os.path.exists(config.dataset):
            os.makedirs(config.dataset)
        self.source_data_path = config.source_data
        self.save_dir = config.dataset

        # lmdb
        self.db = LMDBCreator(config.dataset)

    def parse_and_save_dataset(self):
        # init dataset
        for sample_num, filename in enumerate(tqdm(os.listdir(self.source_data_path), desc='Parsing images...')):
            # read image as ndarray
            image = cv2.imread(self.source_data_path + filename)
            image = cv2.resize(image, dsize=cfg.image_size, interpolation=cv2.INTER_CUBIC)
            image = image

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
    dataparser = DataParser(config=cfg)
    dataparser.parse_and_save_dataset()

