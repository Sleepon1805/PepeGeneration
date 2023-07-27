import os
import lmdb
import pickle
import glob
import zstandard


class LMDBCreator:
    def __init__(self, path, max_size=10e10):
        self.path = path
        self.max_size = int(max_size)

        self.db = self.init_lmdb()
        self.compressor = zstandard.ZstdCompressor()

    def init_lmdb(self):
        if os.path.exists(self.path):
            files = glob.glob(f'{self.path}/*.mdb')
            for f in files:
                os.remove(f)
                print(f'Deleted {f}')
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
