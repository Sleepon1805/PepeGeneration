import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, LearningRateMonitor, RichProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.profilers import AdvancedProfiler
from pytorch_lightning.loggers import TensorBoardLogger

from dataset.dataset import PepeDataset
from model.pepe_generator import PepeGenerator
from config import cfg

if __name__ == '__main__':
    # don't forget to override environment variable HSA_OVERRIDE_GFX_VERSION=10.3.0 (for radeon rx 6700xt)
    torch.set_float32_matmul_precision('high')

    # set num_workers=0 to be able to debug properly
    debug = False

    # dataset
    dataset_name = 'twitch_emotes'
    dataset = PepeDataset(dataset_name, config=cfg, augments=None)
    train_set, val_set = torch.utils.data.random_split(dataset, cfg.dataset_split,
                                                       generator=torch.Generator().manual_seed(42))

    # dataloader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.batch_size, pin_memory=True,
                                               num_workers=0 if debug else 12)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=cfg.batch_size, pin_memory=True,
                                             num_workers=0 if debug else 12)

    # init or load pretrained model
    if cfg.pretrained_ckpt is None:
        model = PepeGenerator(cfg)
    else:
        checkpoint = cfg.pretrained_ckpt
        model = PepeGenerator.load_from_checkpoint(checkpoint, cfg)
        model.config = cfg

    # train the model
    callbacks = [
        RichProgressBar(leave=True),  # progression bar
        EarlyStopping(monitor='val_loss', min_delta=0.0, patience=5, mode='min'),  # early stopping
        ModelCheckpoint(save_top_k=1, monitor='val_loss', save_last=True,
                        filename='{epoch:02d}-{val_loss:.4f}'),  # checkpointing
        ModelSummary(max_depth=2),  # deeper model summary
        LearningRateMonitor(logging_interval='epoch'),  # LR in logger
    ]

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator='auto',
        callbacks=callbacks,
        log_every_n_steps=1,
        # profiler=AdvancedProfiler(filename='profiler'),
        # logger=TensorBoardLogger('lightning_logs', name='', version=1),  # to setup version num
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
