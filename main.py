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
import os

from datetime import datetime

from deepNormalize.training.trainer import DeepNormalizeTrainer
from deepNormalize.utils.initializer import Initializer
from deepNormalize.config.configurations import RunningConfiguration, DeepNormalizeTrainingConfig


def main(config_path: str, running_config: RunningConfiguration):
    init = Initializer(config_path)

    dataset_config, model_config, training_config, variables, logger_config = init.create_configs()

    metrics = init.create_metrics(training_config)

    criterions = init.create_criterions(training_config)

    models = init.create_models(model_config)

    optimizers = init.create_optimizers(training_config, models)

    dataset = init.create_dataset(dataset_config)

    dataloader = init.create_dataloader(dataset, training_config.batch_size)

    train_config = DeepNormalizeTrainingConfig(checkpoint_every=training_config.checkpoint_every,
                                               max_epoch=training_config.max_iterations,
                                               criterion=criterions,
                                               metric=metrics,
                                               model=models,
                                               optimizer=optimizers,
                                               dataloader=dataloader,
                                               running_config=running_config,
                                               variables=variables,
                                               logger_config=logger_config)

    trainer = DeepNormalizeTrainer(train_config, None)

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepNormalize Training with PyTorch and SAMITorch')
    parser.add_argument("--config", help="Path to configuration file.")
    parser.add_argument("--opt-level", type=str, default="O1",
                        help="O0 - FP32 training, 01 - Mixed Precision (recommended), 02 - Almost FP16 Mixed Precision, 03 - FP16 Training.")
    parser.add_argument("--num-workers", default=8, type=int, help="Number of data loading workers (default: 4)")
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument('--sync-batch-norm', action='store_true', default=None, help="Enabling APEX sync Batch Norm.")
    parser.add_argument('--keep-batch-norm-fp32', type=str, default=None)
    parser.add_argument('--loss-scale', type=str, default=None)
    parser.add_argument('--num-gpus', type=int, default=1, help="The number of GPUs on the Node.")
    args = parser.parse_args()

    running_config = RunningConfiguration(dict(vars(args)))

    main(config_path=args.config, running_config=running_config)
