import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from skimage.transform import downscale_local_mean

from config import Paths, Config, HIGHRES_IMAGE_SIZE_MULT
from data.dataset import PepeDataset


class HighResPepeDataset(PepeDataset):
    def __init__(self, dataset_name: str, low_image_size, paths: Paths, augments=None):
        super().__init__(dataset_name, low_image_size * HIGHRES_IMAGE_SIZE_MULT, paths, augments)

    def __getitem__(self, index):
        highres_image, condition = super().__getitem__(index)
        lowres_image = downscale_local_mean(
            highres_image, factors=(1, HIGHRES_IMAGE_SIZE_MULT, HIGHRES_IMAGE_SIZE_MULT)
        )
        lowres_image = torch.from_numpy(lowres_image)

        return highres_image, lowres_image, condition

    @staticmethod
    def show_item(item):
        highres_sample, lowres_sample, condition = item

        highres_image = highres_sample.numpy().transpose((1, 2, 0))
        highres_image = (highres_image + 1) / 2
        lowres_image = lowres_sample.numpy().transpose((1, 2, 0))
        lowres_image = (lowres_image + 1) / 2

        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(highres_image)
        axs[1].imshow(lowres_image)
        fig.suptitle(condition)
        plt.show()
        print(condition)


if __name__ == '__main__':
    cfg = Config()
    dataset = HighResPepeDataset(dataset_name='celeba', low_image_size=cfg.image_size, paths=Paths())

    one_item = dataset[137]
    dataset.show_item(one_item)

    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=8)
    for batch in tqdm(dataloader, desc="Testing data... "):
        assert batch[0].shape[1:] == (3, cfg.image_size * HIGHRES_IMAGE_SIZE_MULT,
                                      cfg.image_size * HIGHRES_IMAGE_SIZE_MULT), batch[0].shape
        assert batch[1].shape[1:] == (3, cfg.image_size, cfg.image_size), batch[1].shape
        assert batch[2].shape[1:] == (40,), batch[2].shape
