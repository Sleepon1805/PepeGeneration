from tqdm import tqdm
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn, TimeRemainingColumn, \
    MofNCompleteColumn
from typing import Iterable


def custom_progress(iterable: Iterable, bar_type: str, desc: str = None):
    if bar_type == 'tqdm':
        return tqdm(iterable, desc=desc)
    elif bar_type == 'rich':
        progress_bar = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn()
        )
        return progress_bar.track(iterable)
    elif bar_type == 'training_rich_bar':
        progress.generating_progress_bar_id = progress.add_task(
            f"[white]Generating {images_batch.shape[0]} images",
            total=len(timesteps)
        )

    else:
        raise ValueError
