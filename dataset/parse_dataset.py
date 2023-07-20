import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pandas as pd

from config import Paths, Config
from lmdb_helper import LMDBCreator
from twitch_emotes_collector import collect_twitch_emotes


class DataParser:
    def __init__(self, paths: Paths, config: Config, dataset_name: str):
        self.dataset_name = dataset_name
        self.source_data_path, self.save_path = self._init_data_paths(paths, self.dataset_name)
        self.db = LMDBCreator(self.save_path)  # lmdb
        self.image_size = (config.image_size, config.image_size)
        self.cond_size = config.condition_size

    @staticmethod
    def _init_data_paths(paths: Paths, dataset_name: str):
        # source data
        if dataset_name == 'celeba':
            source_data_path = paths.celeba_data_path
        elif dataset_name == 'pepe':
            source_data_path = paths.pepe_data_path
            if not os.path.exists(source_data_path):
                collect_twitch_emotes(source_data_path)
        else:
            raise ValueError(f"dataset_name must be 'pepe' or 'celeba', got {dataset_name}")
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

            # parse condition (emoticon name of celeba attributes)
            condition = self._parse_condition(filename)

            # save image
            self.db.write_lmdb_sample(sample_num, (image, condition))

        # save length and keys
        self.db.write_lmdb_metadata(len(os.listdir(self.source_data_path)))

    def _parse_condition(self, filename: str):
        if self.dataset_name == 'pepe':
            if filename.endswith('.png'):
                filename = filename[:-4]
            name = ''.join(i.lower() for i in filename if i.isalpha())  # take only letters in lower case
            enum_letter = (lambda s: ord(s) - 97)  # enumerate lower case letters from 0 to 25
            one_hot_cond = np.zeros((26, self.cond_size))
            for i, letter in enumerate(name):
                one_hot_cond[enum_letter(letter), i] = 1
            return one_hot_cond
        elif self.dataset_name == 'celeba':
            if not hasattr(self, 'attributes_df'):
                csv_path = Path(self.source_data_path).parents[1].joinpath('list_attr_celeba.csv')
                self.attributes_df = pd.read_csv(csv_path)
            attributes = self.attributes_df.loc[self.attributes_df['image_id'] == filename].to_numpy()
            attributes = attributes[0][1:].astype('float32')
            assert attributes.shape == (self.cond_size, )
            return attributes


if __name__ == '__main__':
    dataset_names = [
        'pepe',
        'celeba',
    ]
    for dataset in dataset_names:
        dataparser = DataParser(Paths(), Config(), dataset_name=dataset)
        dataparser.parse_and_save_dataset()

