from torch.utils.tensorboard import SummaryWriter
import os


def get_logger(log_dir) -> SummaryWriter:
    log_dir = os.path.join(log_dir, 'log')
    writer = SummaryWriter(log_dir=log_dir)
    return writer
