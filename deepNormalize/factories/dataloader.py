from typing import Callable
import multiprocessing
import torch
from kerosene.config.trainers import RunConfiguration
from kerosene.config.trainers import TrainerConfiguration
from kerosene.utils.devices import on_single_device
from torch.utils.data import Dataset


class DataloaderFactory(object):
    def __init__(self, test_dataset: Dataset):
        self._test_dataset = test_dataset

    def create_test(self, run_config: RunConfiguration, training_config: TrainerConfiguration,
                    collate_fn: Callable = None):
        devices = run_config.devices

        if not on_single_device(devices):
            torch.distributed.init_process_group(backend='nccl', init_method='env://', rank=run_config.local_rank)
            test_sampler = torch.utils.data.distributed.DistributedSampler(self._test_dataset)

        test_loader = torch.utils.data.DataLoader(dataset=self._test_dataset,
                                                  batch_size=training_config.batch_size,
                                                  shuffle=False if not on_single_device(devices) else True,
                                                  num_workers=run_config.num_workers if run_config.num_workers is not None else
                                                  multiprocessing.cpu_count() // len(
                                                      run_config.devices) if not on_single_device(
                                                      devices) else multiprocessing.cpu_count(),
                                                  sampler=test_sampler if not on_single_device(devices) else None,
                                                  collate_fn=collate_fn,
                                                  pin_memory=torch.cuda.is_available())

        return test_loader
