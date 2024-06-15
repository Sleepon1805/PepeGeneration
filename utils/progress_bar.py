from tqdm import tqdm
from typing import Collection
from typing_extensions import override
from lightning.pytorch.callbacks import (
    TQDMProgressBar,
    RichProgressBar,
)
from rich import progress

from config import USE_RICH_PROGRESS_BAR


class CustomTQDMProgressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__(refresh_rate=1)

    def __call__(self, collection: Collection, desc: str):
        return tqdm(collection, desc=desc, leave=True)

    @override
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.leave = True
        return bar

    @override
    def on_train_epoch_end(self, trainer, *args) -> None:
        super().on_train_epoch_end(trainer, *args)
        metrics = trainer.callback_metrics
        print(f"Epoch {trainer.current_epoch}: " + ", ".join([f'{k}: {v}' for k, v in metrics.items()]) + "\n")


class CustomRichProgressBar(RichProgressBar):
    def __init__(self):
        super().__init__(refresh_rate=1, leave=True)

    def __call__(self, collection: Collection, desc: str):
        if self.progress is None:
            self.progress = progress.Progress(
                progress.SpinnerColumn(),
                progress.TextColumn("[progress.description]{task.description}"),
                progress.BarColumn(),
                progress.MofNCompleteColumn(),
                progress.TimeElapsedColumn(),
                progress.TimeRemainingColumn()
            )
            self.progress.start()

        progress_bar_task = self.progress.add_task(f"[white]{desc}", total=len(collection))
        for item in collection:
            self.progress.update(progress_bar_task, advance=1, visible=True)
            self.refresh()
            yield item

    @override
    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0,
    ) -> None:
        super().on_validation_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)

        # to set validation bar visible
        if not trainer.sanity_checking and self.val_progress_bar_id is not None:
            self.progress.update(self.val_progress_bar_id, advance=0, visible=True)

    @override
    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        super().on_validation_epoch_end(trainer, pl_module)

        # to set validation bar visible
        if self.is_enabled and self.val_progress_bar_id is not None and trainer.state.fn == "fit":
            self.progress.update(self.val_progress_bar_id, advance=0, visible=True)


if USE_RICH_PROGRESS_BAR:
    progress_bar = CustomRichProgressBar()
else:
    progress_bar = CustomTQDMProgressBar()
