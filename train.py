import os
import torch
import torchvision
import torchvision.transforms as T
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from dataset import PepeDataset
from model.pepe_generator import PepeGenerator
from config import cfg


if __name__ == '__main__':
    # don't forget to override HSA_OVERRIDE_GFX_VERSION=10.3.0 as environment variable (for radeon rx 6700xt)
    # tensorboard --logdir ./lightning_logs/
    torch.set_float32_matmul_precision('high')

    # dataset
    print('Loading datasets...')
    dataset = PepeDataset(config=cfg, augments=None)
    train_set, val_set = torch.utils.data.random_split(dataset, cfg.dataset_split,
                                                       generator=torch.Generator().manual_seed(42))

    # dataloader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.batch_size, num_workers=12)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=cfg.batch_size, num_workers=12)

    # init model
    model = PepeGenerator(cfg.image_size[0] * cfg.image_size[1], cfg.diffusion_steps, 3)

    # load checkpoint
    checkpoint = None

    # train the model
    callbacks = [
        # progression bar
        pl.callbacks.RichProgressBar(),
        # accelerator usage
        # pl.callbacks.DeviceStatsMonitor(),
        # early stopping
        EarlyStopping(monitor='val_loss', min_delta=0.0, patience=5, mode='min'),
        # checkpointing
        ModelCheckpoint(save_top_k=1, monitor='val_loss', filename='{epoch:02d}-{val_loss:.4f}'),
    ]

    trainer = pl.Trainer(max_epochs=20,
                         accelerator='auto',
                         devices=1,
                         callbacks=callbacks,
                         log_every_n_steps=1,
                         # profiler="simple",  # running times
                         )

    trainer.fit(model=model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                ckpt_path=checkpoint,  # to start from checkpoint
                )
