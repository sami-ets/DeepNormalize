# -*- coding: utf-8 -*-
# Copyright 2019 Pierre-Luc Delisle. All Rights Reserved.
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import logging
import torch

from visdom import Visdom
from deepNormalize.training.trainer import DeepNormalizeTrainer
from deepNormalize.utils.initializer import Initializer
from deepNormalize.config.configurations import RunningConfiguration, DeepNormalizeTrainerConfig


def main(config_path: str, running_config: RunningConfiguration):
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(4)

    init = Initializer(config_path)

    dataset_config, model_config, training_config, pretraining_config, variables, logger_config, visdom_config = init.create_configs()

    if running_config.is_distributed:
        visdom = Visdom(server=visdom_config.server, port=visdom_config.port,
                        env="DeepNormalize_Autoencoder_GPU_{}".format(running_config.local_rank))

    else:
        visdom = Visdom(server=visdom_config.server, port=visdom_config.port,
                        env="DeepNormalize_Autoencoder")

    init.init_process_group(running_config)

    metrics = init.create_metrics(training_config)

    criterions = init.create_criterions(training_config)

    models = init.create_models(model_config)

    optimizers = init.create_optimizers(training_config, models)

    datasets = init.create_dataset(dataset_config)

    dataloaders = init.create_dataloader(datasets, training_config.batch_size, running_config.num_workers,
                                         dataset_config, is_distributed=running_config.is_distributed)

    train_config = DeepNormalizeTrainerConfig(checkpoint_every=training_config.checkpoint_every,
                                              max_epoch=training_config.max_iterations,
                                              criterion=criterions,
                                              metric=metrics,
                                              model=models,
                                              optimizer=optimizers,
                                              dataloader=dataloaders,
                                              running_config=running_config,
                                              variables=variables,
                                              logger_config=logger_config,
                                              debug=training_config.debug,
                                              pretraining_config=pretraining_config,
                                              visdom=visdom)

    trainer = DeepNormalizeTrainer(train_config, None)

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepNormalize Training with PyTorch and SAMITorch')
    parser.add_argument("--config", help="Path to configuration file.")
    parser.add_argument("--opt-level", type=str, default="O2",
                        help="O0 - FP32 training, O1 - Mixed Precision (recommended), O2 - Almost FP16 Mixed Precision, O3 - FP16 Training.")
    parser.add_argument("--num-workers", default=2, type=int,
                        help="Number of data loading workers for each dataloader object (default: 4)")
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--sync-batch-norm', action='store_true', default=None, help="Enabling APEX sync Batch Norm.")
    parser.add_argument('--keep-batch-norm-fp32', type=str, default=None)
    parser.add_argument('--loss-scale', type=str, default="dynamic")
    parser.add_argument('--num-gpus', type=int, default=1, help="The number of GPUs on the Node.")
    parser.add_argument('--is_distributed', action='store_true', default=False)
    args = parser.parse_args()
    running_config = RunningConfiguration(dict(vars(args)))

    main(config_path=args.config, running_config=running_config)
