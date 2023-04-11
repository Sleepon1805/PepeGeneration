import os

import numpy as np
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from dataset import PepeDataset
from model.pepe_generator import PepeGenerator
from config import cfg

if __name__ == '__main__':
    num_images = 5
    ckpt_path = '/home/sleepon/repos/PepeGenerator/lightning_logs/version_1/checkpoints/epoch=19-val_loss=0.04.ckpt'
    model = PepeGenerator.load_from_checkpoint(ckpt_path)
    images = model.sample(num_images)
    for image in images:
        image = image.cpu().numpy().transpose((1, 2, 0))
        plt.imshow(image)
        plt.show()

