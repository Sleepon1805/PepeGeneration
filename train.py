import sys
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary, LearningRateMonitor, RichProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.profilers import AdvancedProfiler
from lightning.pytorch.loggers import TensorBoardLogger

from dataset.dataset import PepeDataset
from model.pepe_generator import PepeGenerator
from config import Paths, Config

if __name__ == '__main__':
    # don't forget to override environment variable HSA_OVERRIDE_GFX_VERSION=10.3.0 (for radeon rx 6700xt)
    torch.set_float32_matmul_precision('high')
    cfg = Config()

    # set num_workers=0 to be able to debug properly
    debug_mode = hasattr(sys, 'gettrace') and sys.gettrace() is not None
    print(f'Running in debug mode: {str(debug_mode)}')

    # dataset
    dataset = PepeDataset(cfg.dataset_name, paths=Paths(), augments=None)
    train_set, val_set = torch.utils.data.random_split(dataset, cfg.dataset_split,
                                                       generator=torch.Generator().manual_seed(42))

    # dataloader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.batch_size, pin_memory=True,
                                               num_workers=0 if debug_mode else 12)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=cfg.batch_size, pin_memory=True,
                                             num_workers=0 if debug_mode else 12)

    # init or load pretrained model
    if cfg.pretrained_ckpt is None:
        model = PepeGenerator(cfg)
    else:
        checkpoint = cfg.pretrained_ckpt
        model = PepeGenerator.load_from_checkpoint(checkpoint, config=cfg)
        model.config = cfg
    # model = torch.compile(model)  # FIXME

    # logger
    logger = TensorBoardLogger('lightning_logs', name='', log_graph=False, default_hp_metric=False,
                               # version=0,
                               )

    # train the model
    callbacks = [
        RichProgressBar(leave=True),  # progression bar
        EarlyStopping(monitor='val_loss', min_delta=0.0, patience=5, mode='min'),  # early stopping
        ModelCheckpoint(save_top_k=1, monitor='val_loss', save_last=True,
                        filename='{epoch:02d}-{val_loss:.4f}'),  # checkpointing
        ModelSummary(max_depth=2),  # deeper model summary
        LearningRateMonitor(logging_interval='epoch'),  # LR in logger
    ]

    trainer = L.Trainer(
        max_epochs=50,
        accelerator='auto',
        callbacks=callbacks,
        log_every_n_steps=1,
        # profiler=AdvancedProfiler(filename='profiler'),
        logger=logger,
        # num_sanity_val_steps=0,
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        # ckpt_path='lightning_logs/version_0/checkpoints/last.ckpt'
    )
