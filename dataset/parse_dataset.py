import os
import cv2
from tqdm import tqdm

from config import Paths, Config
from lmdb_helper import LMDBCreator
from twitch_emotes_collector import collect_twitch_emotes


class DataParser:
    def __init__(self, paths: Paths, config: Config, dataset_name: str):
        self.dataset_name = dataset_name
        self.source_data_path, self.save_path = self._init_data_paths(paths, self.dataset_name)
        self.db = LMDBCreator(self.save_path)  # lmdb
        self.image_size = (config.image_size, config.image_size)

    @staticmethod
    def _init_data_paths(paths: Paths, dataset_name: str):
        # source data
        if dataset_name == 'pepe':
            source_data_path = paths.pepe_data_path
        elif dataset_name == 'celeba':
            source_data_path = paths.celeba_data_path
        elif dataset_name == 'twitch_emotes':
            source_data_path = paths.twitch_emotes_data_path
            if not os.path.exists(source_data_path):
                collect_twitch_emotes(source_data_path)
        else:
            raise ValueError(f"dataset_name must be 'pepe', 'celeba' or 'twitch_emotes', got {dataset_name}")
        assert os.path.exists(source_data_path), \
            f'Given path for source images {source_data_path} does not exist! Change it in config.py.'

        # parsed data
        save_path = paths.parsed_datasets + dataset_name
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        return source_data_path, save_path

    def parse_and_save_dataset(self):
        # init dataset
        for sample_num, filename in enumerate(tqdm(os.listdir(self.source_data_path),
                                                   desc=f'Parsing {self.dataset_name} images...')):
            # read image as ndarray
            image = cv2.imread(self.source_data_path + filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, dsize=self.image_size, interpolation=cv2.INTER_CUBIC)

            # save image
            self.db.write_lmdb_sample(sample_num, image)

        # save length and keys
        self.db.write_lmdb_metadata(len(os.listdir(self.source_data_path)))


if __name__ == '__main__':
    dataset_names = [
        'pepe',
        'celeba',
        'twitch_emotes'
    ]
    for dataset in dataset_names:
        dataparser = DataParser(Paths(), Config(), dataset_name=dataset)
        dataparser.parse_and_save_dataset()

