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

import logging
import multiprocessing
import os
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from kerosene.config.parsers import YamlConfigurationParser
from kerosene.config.trainers import RunConfiguration
from kerosene.metrics.gauges import AverageGauge
from scipy.spatial.distance import directed_hausdorff
from deepNormalize.factories.dataloader import DataloaderFactory
from kerosene.events.handlers.checkpoints import ModelCheckpointIfBetter
from kerosene.events.handlers.console import PrintTrainingStatus
from kerosene.loggers.visdom.config import VisdomConfiguration
from kerosene.loggers.visdom.visdom import VisdomLogger
from kerosene.training.trainers import ModelTrainerFactory
from samitorch.inputs.utils import sample_collate
from torch.utils.data import DataLoader
from kerosene.utils.tensors import to_onehot, flatten
from torchvision.transforms import transforms
from deepNormalize.config.parsers import ArgsParserFactory, ArgsParserType, DatasetConfigurationParser
from deepNormalize.factories.customCriterionFactory import CustomCriterionFactory
from deepNormalize.factories.customModelFactory import CustomModelFactory

from deepNormalize.inputs.datasets import iSEGPatchDatasetFactory, MRBrainSPatchDatasetFactory
from samitorch.inputs.transformers import ToNumpyArray
from samitorch.inputs.utils import patch_collate
from kerosene.nn.functional import js_div

ISEG_ID = 0
MRBRAINS_ID = 1

cudnn.benchmark = True
cudnn.enabled = True

if __name__ == '__main__':
    segmenter_path = ""
    generator_path = ""

    generator_loss_test_gauge = AverageGauge()
    segmenter_loss_test_gauge = AverageGauge()
    segmenter_metric_test_gauge = AverageGauge()
    class_hausdorff_distance_gauge = AverageGauge()
    class_dice_gauge = AverageGauge()
    confusion_matrix_gauge = AverageGauge()
    js_div_inputs_gauge = AverageGauge()
    js_div_gen_gauge = AverageGauge()

    # Basic settings
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(multiprocessing.cpu_count())
    torch.set_num_interop_threads(multiprocessing.cpu_count())
    args = ArgsParserFactory.create_parser(ArgsParserType.MODEL_TRAINING).parse_args()

    # Create configurations.
    run_config = RunConfiguration(args.use_amp, args.amp_opt_level, args.local_rank, args.num_workers)
    model_trainer_configs, training_config = YamlConfigurationParser.parse(args.config_file)
    dataset_config = DatasetConfigurationParser().parse(args.config_file)
    config_html = [training_config.to_html(), list(map(lambda config: config.to_html(), dataset_config)),
                   list(map(lambda config: config.to_html(), model_trainer_configs))]

    iSEG_test = iSEGPatchDatasetFactory.create_test(source_dir=dataset_config[0].path,
                                                    target_dir=dataset_config[0].path + "/label",
                                                    modality=args.modality,
                                                    dataset_id=0,
                                                    patch_size=dataset_config[0].test_patch_size,
                                                    step=dataset_config[0].test_patch_step,
                                                    keep_centered_on_foreground=False)

    MRBrainS_test = MRBrainSPatchDatasetFactory.create_test(source_dir=dataset_config[1].path,
                                                            target_dir=dataset_config[1].path,
                                                            modality=args.modality,
                                                            dataset_id=1,
                                                            patch_size=dataset_config[1].test_patch_size,
                                                            step=dataset_config[1].test_patch_step,
                                                            keep_centered_on_foreground=False)

    # Concat datasets.
    if len(dataset_config) == 2:
        test_set = torch.utils.data.ConcatDataset((iSEG_test, MRBrainS_test))
    else:
        test_set = iSEG_test if iSEG_test is not None else MRBrainS_test

    # Create loaders.
    test_loader = DataloaderFactory(test_set).create_test(run_config,
                                                          training_config,
                                                          collate_fn=patch_collate)

    model_trainer_factory = ModelTrainerFactory(model_factory=CustomModelFactory())

    model_trainers = list(map(lambda config: model_trainer_factory.create(config, run_config), model_trainer_configs))

    # Initialize the loggers.
    visdom_config = VisdomConfiguration.from_yml(args.config_file, "visdom")
    visdom_logger = VisdomLogger(visdom_config)

    generator = model_trainers[0]
    generator.model.load_state_dict(torch.load(generator_path))
    generator.model = generator.model.cuda()
    segmenter = model_trainers[1]
    segmenter.model.load_state_dict(torch.load(segmenter_path))
    segmenter.model = segmenter.model.cuda()

    transforms_ = transforms.Compose([ToNumpyArray()])

    gen_pred_np = list()
    seg_pred_np = list()
    # Train with the training strategy.

    with torch.no_grad():
        for current_test_batch, (inputs, target) in enumerate(test_loader):

            inputs = [single_input.to("cuda:0", non_blocking=True) for single_input in
                      inputs] if isinstance(inputs, list) else inputs.to("cuda:0", non_blocking=True)

            target = [single_target.to("cuda:0", non_blocking=True) for single_target in
                      target] if isinstance(target, list) else target.to("cuda:0", non_blocking=True)

            gen_pred = generator.forward(inputs)
            gen_loss = generator.compute_loss(gen_pred, target[0])
            generator_loss_test_gauge.update(float(gen_loss.loss))

            seg_pred = segmenter.forward(gen_pred)
            seg_loss = segmenter.compute_loss(torch.nn.functional.softmax(seg_pred, dim=1),
                                              to_onehot(torch.squeeze(target[0], dim=1).long(),
                                                        num_classes=4))
            segmenter_loss_test_gauge.update(float(seg_loss.mean()))

            metric = segmenter.compute_metric(torch.nn.functional.softmax(seg_pred, dim=1),
                                              torch.squeeze(target[0], dim=1).long())
            segmenter_metric_test_gauge.update(float(metric.mean()))

            seg_pred_ = to_onehot(torch.argmax(torch.nn.functional.softmax(seg_pred, dim=1), dim=1), num_classes=4)
            target_ = to_onehot(torch.squeeze(target[0], dim=1).long(), num_classes=4)

            distances = np.zeros((4,))
            for channel in range(seg_pred_.size(1)):
                distances[channel] = (
                                             directed_hausdorff(
                                                 flatten(seg_pred_[:, channel, ...]).cpu().detach().numpy(),
                                                 flatten(target_[:, channel, ...]).cpu().detach().numpy())[0] +
                                             directed_hausdorff(
                                                 flatten(target_[:, channel, ...]).cpu().detach().numpy(),
                                                 flatten(seg_pred_[:, channel, ...]).cpu().detach().numpy())[0]) / 2.0

            class_hausdorff_distance_gauge.update(distances)
            class_dice_gauge.update(metric.numpy())

            confusion_matrix_gauge.update((
                to_onehot(torch.argmax(torch.nn.functional.softmax(seg_pred, dim=1), dim=1, keepdim=False),
                          num_classes=4),
                torch.squeeze(target[0].long(), dim=1)))

            js_div_inputs_gauge.update(js_div(inputs))
            js_div_gen_gauge.update(js_div(gen_pred))

    print("Jensen-Shannon distance on generated : {}".format(js_div_gen_gauge.compute()))
    print("Jensen-Shannon distance on inputs : {}".format(js_div_inputs_gauge.compute()))
    print("Dice over test set : {}".format(segmenter_metric_test_gauge.compute()))
