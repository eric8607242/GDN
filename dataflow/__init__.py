import sys

def get_dataloader(dataset_name, dataset_path, input_size, batch_size, num_workers, train_portion, slide_win, slide_stride):
    dataloader_builder = getattr(sys.modules[__name__], f"get_{dataset_name}_dataloader")

    return dataloader_builder(dataset_path, input_size, batch_size, num_workers, train_portion, slide_win, slide_stride)


# Import customizing module
from .wadi import get_wadi_dataloader
