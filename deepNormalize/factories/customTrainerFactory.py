from enum import Enum
from kerosene.events import MonitorMode
from kerosene.events.handlers.checkpoints import Checkpoint
from kerosene.events.handlers.console import PrintTrainingStatus, PrintMonitors
from kerosene.events.handlers.visdom import PlotMonitors, PlotLR, PlotCustomVariables, PlotAvgGradientPerLayer
from kerosene.loggers.visdom import PlotType
from kerosene.training.events import Event

from deepNormalize.events.handlers.handlers import PlotCustomLinePlotWithLegend, PlotCustomLoss
from deepNormalize.training.dcgan import DCGANTrainer
from deepNormalize.training.dual_unet import DualUNetTrainer
from deepNormalize.training.lsgan import LSGANTrainer
from deepNormalize.training.resnet import ResNetTrainer
from deepNormalize.training.unet import UNetTrainer
from deepNormalize.training.wgan import WGANTrainer


class TrainerType(Enum):
    WGAN = "WGAN"
    ResNet = "ResNet"
    DUALUNET = "DualUNet"
    UNET = "UNet"
    DCGAN = "DCGAN"
    LSGAN = "LSGAN"

    def __str__(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        else:
            return False


class TrainerFactory(object):

    def __init__(self, architecture):
        self._trainer = architecture

    def create(self, training_config, model_trainers, dataloaders, reconstruction_datasets,
               normalized_reconstructors, input_reconstructors, segmentation_reconstructors,
               augmented_input_reconstructors, gt_reconstructors, run_config, dataset_configs, save_folder,
               visdom_logger):
        if self._trainer == TrainerType.WGAN:
            trainer = WGANTrainer(training_config, model_trainers, dataloaders[0], dataloaders[1],
                                  dataloaders[2],
                                  reconstruction_datasets, normalized_reconstructors, input_reconstructors,
                                  segmentation_reconstructors, augmented_input_reconstructors,
                                  gt_reconstructors,
                                  run_config, dataset_configs, save_folder) \
                .with_event_handler(PrintTrainingStatus(every=25), Event.ON_BATCH_END) \
                .with_event_handler(PrintMonitors(every=25), Event.ON_BATCH_END) \
                .with_event_handler(PlotMonitors(visdom_logger), Event.ON_EPOCH_END) \
                .with_event_handler(PlotLR(visdom_logger), Event.ON_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Training Generated Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Training Generated Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Validation Generated Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Validation Generated Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_VALID_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Test Generated Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Test Generated Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Training Input Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Training Input Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Validation Input Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Validation Input Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_VALID_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Test Input Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Test Input Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Training Segmented Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Segmented Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Validation Segmented Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Validation Segmented Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_VALID_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Test Segmented Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Test Segmented Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Training Segmentation Ground Truth Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Training Ground Truth Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Validation Segmentation Ground Truth Batch Process {}".format(
                                        run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Validation Ground Truth Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_VALID_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Test Segmentation Ground Truth Batch Process {}".format(
                                        run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Test Ground Truth Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Training Label Map Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Training Label Map Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Validation Label Map Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Validation Label Map Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_VALID_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Test Label Map Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Test Label Map Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Generated Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "Generated Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Background Generated Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "Background Generated Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "CSF Generated Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "CSF Generated Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "GM Generated Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "GM Generated Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "WM Generated Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "WM Generated Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "Inputs Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Background Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "Background Input Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "CSF Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "CSF Input Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "GM Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "GM Input Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "WM Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "WM Input Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Pie Plot", PlotType.PIE_PLOT,
                                                        params={"opts": {"title": "Classification hit per classes",
                                                                         "legend": list(map(lambda key: key,
                                                                                            dataset_configs.keys())) + [
                                                                                       "Fake Class"]}},
                                                        every=25), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Pie Plot True", PlotType.PIE_PLOT,
                                                        params={"opts": {"title": "Batch data distribution",
                                                                         "legend": list(map(lambda key: key,
                                                                                            dataset_configs.keys())) + [
                                                                                       "Fake Class"]}},
                                                        every=25), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Mean Hausdorff Distance", PlotType.LINE_PLOT,
                                                        params={"opts": {"title": "Mean Hausdorff Distance",
                                                                         "legend": ["Test"]}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Metric Table", PlotType.TEXT_PLOT,
                                                        params={"opts": {"title": "Metric Table"}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Per-Dataset Metric Table", PlotType.TEXT_PLOT,
                                                        params={"opts": {"title": "Per-Dataset Metric Table"}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Jensen-Shannon Table", PlotType.TEXT_PLOT,
                                                        params={"opts": {"title": "Jensen-Shannon Divergence"}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Confusion Matrix", PlotType.HEATMAP_PLOT,
                                                        params={
                                                            "opts": {"columnnames": ["VM", "GM", "CSF", "Background"],
                                                                     "rownames": ["Background", "CSF", "GM", "WM"],
                                                                     "title": "Confusion Matrix"}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "iSEG Confusion Matrix", PlotType.HEATMAP_PLOT,
                                                        params={
                                                            "opts": {"columnnames": ["VM", "GM", "CSF", "Background"],
                                                                     "rownames": ["Background", "CSF", "GM", "WM"],
                                                                     "title": "iSEG Confusion Matrix"}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "MRBrainS Confusion Matrix", PlotType.HEATMAP_PLOT,
                                    params={"opts": {"columnnames": ["VM", "GM", "CSF", "Background"],
                                                     "rownames": ["Background", "CSF", "GM", "WM"],
                                                     "title": "MRBrainS Confusion Matrix"}},
                                    every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "ABIDE Confusion Matrix", PlotType.HEATMAP_PLOT,
                                                        params={
                                                            "opts": {"columnnames": ["VM", "GM", "CSF", "Background"],
                                                                     "rownames": ["Background", "CSF", "GM", "WM"],
                                                                     "title": "ABIDE Confusion Matrix"}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Discriminator Confusion Matrix", PlotType.HEATMAP_PLOT,
                                    params={"opts": {
                                        "columnnames": ["Generated"] + list(reversed(list(dataset_configs.keys()))),
                                        "rownames": list(dataset_configs.keys()) + ["Generated"],
                                        "title": "Discriminator Confusion Matrix"}},
                                    every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Discriminator Confusion Matrix Training", PlotType.HEATMAP_PLOT,
                                    params={"opts": {
                                        "columnnames": ["Generated"] + list(reversed(list(dataset_configs.keys()))),
                                        "rownames": list(dataset_configs.keys()) + ["Generated"],
                                        "title": "Discriminator Confusion Matrix Training"}},
                                    every=1), Event.ON_TRAIN_EPOCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Runtime", PlotType.TEXT_PLOT,
                                                        params={"opts": {"title": "Runtime"}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(PlotCustomLoss(visdom_logger, "D(G(X)) | X", every=1), Event.ON_EPOCH_END) \
                .with_event_handler(PlotCustomLoss(visdom_logger, "Wasserstein Distance", every=1),
                                    Event.ON_EPOCH_END) \
                .with_event_handler(PlotCustomLoss(visdom_logger, "Total Loss", every=1), Event.ON_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger, "Jensen-Shannon Divergence", every=1,
                                             params={"title": "Jensen-Shannon Divergence on test data per Epoch",
                                                     "legend": ["Inputs", "Normalized"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger, "Dice score per class per epoch", every=1,
                                             params={"title": "Dice score on test patches per class per epoch",
                                                     "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger,
                                             "Hausdorff Distance per class per epoch on reconstructed iSEG image",
                                             every=1,
                                             params={
                                                 "title": "Hausdorff Distance score per class per epoch on reconstructed iSEG image",
                                                 "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger,
                                             "Hausdorff Distance per class per epoch on reconstructed MRBrainS image",
                                             every=1,
                                             params={
                                                 "title": "Hausdorff Distance per class per epoch on reconstructed MRBrainS image",
                                                 "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger,
                                             "Hausdorff Distance per class per epoch on reconstructed ABIDE image",
                                             every=1,
                                             params={
                                                 "title": "Hausdorff Distance per class per epoch on reconstructed ABIDE image",
                                                 "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger,
                                             "Dice score per class per epoch on reconstructed iSEG image",
                                             every=1,
                                             params={
                                                 "title": "Dice score per class per epoch on reconstructed iSEG image",
                                                 "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger,
                                             "Dice score per class per epoch on reconstructed MRBrainS image", every=1,
                                             params={
                                                 "title": "Dice score per class per epoch on reconstructed MRBrainS image",
                                                 "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger,
                                             "Dice score per class per epoch on reconstructed ABIDE image",
                                             every=1,
                                             params={
                                                 "title": "Dice score per class per epoch on reconstructed ABIDE image",
                                                 "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Input iSEG Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Input iSEG Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Normalized iSEG Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Normalized iSEG Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Segmented iSEG Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Segmented iSEG Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Ground Truth iSEG Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Ground Truth iSEG Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Initial Noise iSEG Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Initial Noise iSEG Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Noise iSEG After Normalization", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Noise iSEG After Normalization"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Input MRBrainS Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Input MRBrainS Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Normalized MRBrainS Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Normalized MRBrainS Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Segmented MRBrainS Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Segmented MRBrainS Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Ground Truth MRBrainS Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Ground Truth MRBrainS Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Initial Noise MRBrainS Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Initial Noise MRBrainS Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Noise MRBrainS After Normalization",
                                    PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Noise MRBrainS After Normalization"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Input ABIDE Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Input ABIDE Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Normalized ABIDE Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Normalized ABIDE Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Ground Truth ABIDE Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Ground Truth ABIDE Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Segmented ABIDE Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Segmented ABIDE Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Conv1 FM", PlotType.IMAGES_PLOT,
                                    params={"nrow": 8, "opts": {"store_history": True,
                                                                "title": "Conv1 FM"}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Layer1 FM", PlotType.IMAGES_PLOT,
                                    params={"nrow": 8, "opts": {"store_history": True,
                                                                "title": "Layer1 FM"}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Layer2 FM", PlotType.IMAGES_PLOT,
                                    params={"nrow": 12, "opts": {"store_history": True,
                                                                 "title": "Layer2 FM"}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Layer3 FM", PlotType.IMAGES_PLOT,
                                    params={"nrow": 16, "opts": {"store_history": True,
                                                                 "title": "Layer3 FM"}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Per-Dataset Histograms", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Images Histograms", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True}}, every=5), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                Checkpoint(save_folder, monitor_fn=lambda model_trainer: model_trainer.valid_loss, delta=0.01,
                           mode=MonitorMode.MIN), Event.ON_EPOCH_END) \
                .with_event_handler(PlotAvgGradientPerLayer(visdom_logger, every=25), Event.ON_TRAIN_BATCH_END)

            return trainer

        elif self._trainer == TrainerType.ResNet:
            trainer = ResNetTrainer(training_config, model_trainers, dataloaders[0], dataloaders[1],
                                    dataloaders[2],
                                    reconstruction_datasets, normalized_reconstructors, input_reconstructors,
                                    segmentation_reconstructors, augmented_input_reconstructors,
                                    gt_reconstructors,
                                    run_config, dataset_configs, save_folder) \
                .with_event_handler(PrintTrainingStatus(every=25), Event.ON_BATCH_END) \
                .with_event_handler(PrintMonitors(every=25), Event.ON_BATCH_END) \
                .with_event_handler(PlotMonitors(visdom_logger), Event.ON_EPOCH_END) \
                .with_event_handler(PlotLR(visdom_logger), Event.ON_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Training Generated Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Training Generated Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Validation Generated Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Validation Generated Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_VALID_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Test Generated Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Test Generated Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Training Input Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Training Input Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Validation Input Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Validation Input Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_VALID_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Test Input Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Test Input Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Training Segmented Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Segmented Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Validation Segmented Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Validation Segmented Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_VALID_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Test Segmented Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Test Segmented Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Training Segmentation Ground Truth Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Training Ground Truth Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Validation Segmentation Ground Truth Batch Process {}".format(
                                        run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Validation Ground Truth Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_VALID_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Test Segmentation Ground Truth Batch Process {}".format(
                                        run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Test Ground Truth Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Training Label Map Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Training Label Map Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Validation Label Map Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Validation Label Map Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_VALID_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Test Label Map Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Test Label Map Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Generated Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "Generated Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Background Generated Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "Background Generated Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "CSF Generated Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "CSF Generated Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "GM Generated Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "GM Generated Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "WM Generated Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "WM Generated Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "Inputs Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Background Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "Background Input Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "CSF Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "CSF Input Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "GM Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "GM Input Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "WM Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "WM Input Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Pie Plot", PlotType.PIE_PLOT,
                                                        params={"opts": {"title": "Classification hit per classes",
                                                                         "legend": list(map(lambda key: key,
                                                                                            dataset_configs.keys())) + [
                                                                                       "Fake Class"]}},
                                                        every=25), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Pie Plot True", PlotType.PIE_PLOT,
                                                        params={"opts": {"title": "Batch data distribution",
                                                                         "legend": list(map(lambda key: key,
                                                                                            dataset_configs.keys())) + [
                                                                                       "Fake Class"]}},
                                                        every=25), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Mean Hausdorff Distance", PlotType.LINE_PLOT,
                                                        params={"opts": {"title": "Mean Hausdorff Distance",
                                                                         "legend": ["Test"]}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Metric Table", PlotType.TEXT_PLOT,
                                                        params={"opts": {"title": "Metric Table"}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Per-Dataset Metric Table", PlotType.TEXT_PLOT,
                                                        params={"opts": {"title": "Per-Dataset Metric Table"}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Jensen-Shannon Table", PlotType.TEXT_PLOT,
                                                        params={"opts": {"title": "Jensen-Shannon Divergence"}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Confusion Matrix", PlotType.HEATMAP_PLOT,
                                                        params={
                                                            "opts": {"columnnames": ["VM", "GM", "CSF", "Background"],
                                                                     "rownames": ["Background", "CSF", "GM", "WM"],
                                                                     "title": "Confusion Matrix"}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "iSEG Confusion Matrix", PlotType.HEATMAP_PLOT,
                                                        params={
                                                            "opts": {"columnnames": ["VM", "GM", "CSF", "Background"],
                                                                     "rownames": ["Background", "CSF", "GM", "WM"],
                                                                     "title": "iSEG Confusion Matrix"}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "MRBrainS Confusion Matrix", PlotType.HEATMAP_PLOT,
                                    params={"opts": {"columnnames": ["VM", "GM", "CSF", "Background"],
                                                     "rownames": ["Background", "CSF", "GM", "WM"],
                                                     "title": "MRBrainS Confusion Matrix"}},
                                    every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "ABIDE Confusion Matrix", PlotType.HEATMAP_PLOT,
                                                        params={
                                                            "opts": {"columnnames": ["VM", "GM", "CSF", "Background"],
                                                                     "rownames": ["Background", "CSF", "GM", "WM"],
                                                                     "title": "ABIDE Confusion Matrix"}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Discriminator Confusion Matrix", PlotType.HEATMAP_PLOT,
                                    params={"opts": {
                                        "columnnames": ["Generated"] + list(reversed(list(dataset_configs.keys()))),
                                        "rownames": list(dataset_configs.keys()) + ["Generated"],
                                        "title": "Discriminator Confusion Matrix"}},
                                    every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Discriminator Confusion Matrix Training", PlotType.HEATMAP_PLOT,
                                    params={"opts": {
                                        "columnnames": ["Generated"] + list(reversed(list(dataset_configs.keys()))),
                                        "rownames": list(dataset_configs.keys()) + ["Generated"],
                                        "title": "Discriminator Confusion Matrix Training"}},
                                    every=1), Event.ON_TRAIN_EPOCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Runtime", PlotType.TEXT_PLOT,
                                                        params={"opts": {"title": "Runtime"}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(PlotCustomLoss(visdom_logger, "D(G(X)) | X", every=1), Event.ON_EPOCH_END) \
                .with_event_handler(PlotCustomLoss(visdom_logger, "Discriminator Loss", every=1),
                                    Event.ON_EPOCH_END) \
                .with_event_handler(PlotCustomLoss(visdom_logger, "Total Loss", every=1), Event.ON_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger, "Jensen-Shannon Divergence", every=1,
                                             params={"title": "Jensen-Shannon Divergence on test data per Epoch",
                                                     "legend": ["Inputs", "Normalized"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger, "Per Dataset Mean Hausdorff Distance", every=1,
                                             params={"title": "Per Dataset Mean Hausdorff Distance",
                                                     "legend": list(dataset_configs.keys())}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger, "Dice score per class per epoch", every=1,
                                             params={"title": "Dice score on test patches per class per epoch",
                                                     "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger,
                                             "Hausdorff Distance per class per epoch on reconstructed iSEG image",
                                             every=1,
                                             params={
                                                 "title": "Hausdorff Distance score per class per epoch on reconstructed iSEG image",
                                                 "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger,
                                             "Hausdorff Distance per class per epoch on reconstructed MRBrainS image",
                                             every=1,
                                             params={
                                                 "title": "Hausdorff Distance per class per epoch on reconstructed MRBrainS image",
                                                 "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger,
                                             "Hausdorff Distance per class per epoch on reconstructed ABIDE image",
                                             every=1,
                                             params={
                                                 "title": "Hausdorff Distance per class per epoch on reconstructed ABIDE image",
                                                 "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger,
                                             "Dice score per class per epoch on reconstructed iSEG image",
                                             every=1,
                                             params={
                                                 "title": "Dice score per class per epoch on reconstructed iSEG image",
                                                 "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger,
                                             "Dice score per class per epoch on reconstructed MRBrainS image", every=1,
                                             params={
                                                 "title": "Dice score per class per epoch on reconstructed MRBrainS image",
                                                 "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger,
                                             "Dice score per class per epoch on reconstructed ABIDE image",
                                             every=1,
                                             params={
                                                 "title": "Dice score per class per epoch on reconstructed ABIDE image",
                                                 "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Input iSEG Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Input iSEG Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Normalized iSEG Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Normalized iSEG Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Segmented iSEG Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Segmented iSEG Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Ground Truth iSEG Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Ground Truth iSEG Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Initial Noise iSEG Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Initial Noise iSEG Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Noise iSEG After Normalization", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Noise iSEG After Normalization"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Input MRBrainS Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Input MRBrainS Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Normalized MRBrainS Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Normalized MRBrainS Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Segmented MRBrainS Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Segmented MRBrainS Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Ground Truth MRBrainS Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Ground Truth MRBrainS Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Initial Noise MRBrainS Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Initial Noise MRBrainS Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Noise MRBrainS After Normalization",
                                    PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Noise MRBrainS After Normalization"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Input ABIDE Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Input ABIDE Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Normalized ABIDE Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Normalized ABIDE Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Ground Truth ABIDE Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Ground Truth ABIDE Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Segmented ABIDE Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Segmented ABIDE Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Conv1 FM", PlotType.IMAGES_PLOT,
                                    params={"nrow": 8, "opts": {"store_history": True,
                                                                "title": "Conv1 FM"}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Layer1 FM", PlotType.IMAGES_PLOT,
                                    params={"nrow": 8, "opts": {"store_history": True,
                                                                "title": "Layer1 FM"}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Layer2 FM", PlotType.IMAGES_PLOT,
                                    params={"nrow": 12, "opts": {"store_history": True,
                                                                 "title": "Layer2 FM"}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Layer3 FM", PlotType.IMAGES_PLOT,
                                    params={"nrow": 16, "opts": {"store_history": True,
                                                                 "title": "Layer3 FM"}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Per-Dataset Histograms", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Images Histograms", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True}}, every=5), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                Checkpoint(save_folder, monitor_fn=lambda model_trainer: model_trainer.valid_loss, delta=0.01,
                           mode=MonitorMode.MIN), Event.ON_EPOCH_END) \
                .with_event_handler(PlotAvgGradientPerLayer(visdom_logger, every=25), Event.ON_TRAIN_BATCH_END)

            return trainer

        elif self._trainer == TrainerType.DCGAN:
            trainer = DCGANTrainer(training_config, model_trainers, dataloaders[0], dataloaders[1],
                                   dataloaders[2],
                                   reconstruction_datasets, normalized_reconstructors, input_reconstructors,
                                   segmentation_reconstructors, augmented_input_reconstructors,
                                   gt_reconstructors,
                                   run_config, dataset_configs, save_folder) \
                .with_event_handler(PrintTrainingStatus(every=25), Event.ON_BATCH_END) \
                .with_event_handler(PrintMonitors(every=25), Event.ON_BATCH_END) \
                .with_event_handler(PlotMonitors(visdom_logger), Event.ON_EPOCH_END) \
                .with_event_handler(PlotLR(visdom_logger), Event.ON_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Training Generated Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Training Generated Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Validation Generated Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Validation Generated Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_VALID_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Test Generated Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Test Generated Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Training Input Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Training Input Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Validation Input Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Validation Input Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_VALID_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Test Input Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Test Input Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Training Segmented Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Segmented Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Validation Segmented Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Validation Segmented Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_VALID_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Test Segmented Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Test Segmented Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Training Segmentation Ground Truth Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Training Ground Truth Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Validation Segmentation Ground Truth Batch Process {}".format(
                                        run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Validation Ground Truth Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_VALID_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Test Segmentation Ground Truth Batch Process {}".format(
                                        run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Test Ground Truth Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Training Label Map Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Training Label Map Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Validation Label Map Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Validation Label Map Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_VALID_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Test Label Map Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Test Label Map Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Generated Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "Generated Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Background Generated Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "Background Generated Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "CSF Generated Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "CSF Generated Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "GM Generated Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "GM Generated Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "WM Generated Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "WM Generated Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "Inputs Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Background Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "Background Input Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "CSF Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "CSF Input Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "GM Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "GM Input Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "WM Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "WM Input Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Pie Plot", PlotType.PIE_PLOT,
                                                        params={"opts": {"title": "Classification hit per classes",
                                                                         "legend": list(map(lambda key: key,
                                                                                            dataset_configs.keys())) + [
                                                                                       "Fake Class"]}},
                                                        every=25), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Pie Plot True", PlotType.PIE_PLOT,
                                                        params={"opts": {"title": "Batch data distribution",
                                                                         "legend": list(map(lambda key: key,
                                                                                            dataset_configs.keys())) + [
                                                                                       "Fake Class"]}},
                                                        every=25), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Mean Hausdorff Distance", PlotType.LINE_PLOT,
                                                        params={"opts": {"title": "Mean Hausdorff Distance",
                                                                         "legend": ["Test"]}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Metric Table", PlotType.TEXT_PLOT,
                                                        params={"opts": {"title": "Metric Table"}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Per-Dataset Metric Table", PlotType.TEXT_PLOT,
                                                        params={"opts": {"title": "Per-Dataset Metric Table"}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Jensen-Shannon Table", PlotType.TEXT_PLOT,
                                                        params={"opts": {"title": "Jensen-Shannon Divergence"}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Confusion Matrix", PlotType.HEATMAP_PLOT,
                                                        params={
                                                            "opts": {"columnnames": ["VM", "GM", "CSF", "Background"],
                                                                     "rownames": ["Background", "CSF", "GM", "WM"],
                                                                     "title": "Confusion Matrix"}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "iSEG Confusion Matrix", PlotType.HEATMAP_PLOT,
                                                        params={
                                                            "opts": {"columnnames": ["VM", "GM", "CSF", "Background"],
                                                                     "rownames": ["Background", "CSF", "GM", "WM"],
                                                                     "title": "iSEG Confusion Matrix"}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "MRBrainS Confusion Matrix", PlotType.HEATMAP_PLOT,
                                    params={"opts": {"columnnames": ["VM", "GM", "CSF", "Background"],
                                                     "rownames": ["Background", "CSF", "GM", "WM"],
                                                     "title": "MRBrainS Confusion Matrix"}},
                                    every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "ABIDE Confusion Matrix", PlotType.HEATMAP_PLOT,
                                                        params={
                                                            "opts": {"columnnames": ["VM", "GM", "CSF", "Background"],
                                                                     "rownames": ["Background", "CSF", "GM", "WM"],
                                                                     "title": "ABIDE Confusion Matrix"}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Discriminator Confusion Matrix", PlotType.HEATMAP_PLOT,
                                    params={"opts": {
                                        "columnnames": ["Generated"] + list(reversed(list(dataset_configs.keys()))),
                                        "rownames": list(dataset_configs.keys()) + ["Generated"],
                                        "title": "Discriminator Confusion Matrix"}},
                                    every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Discriminator Confusion Matrix Training", PlotType.HEATMAP_PLOT,
                                    params={"opts": {
                                        "columnnames": ["Generated"] + list(reversed(list(dataset_configs.keys()))),
                                        "rownames": list(dataset_configs.keys()) + ["Generated"],
                                        "title": "Discriminator Confusion Matrix Training"}},
                                    every=1), Event.ON_TRAIN_EPOCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Runtime", PlotType.TEXT_PLOT,
                                                        params={"opts": {"title": "Runtime"}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(PlotCustomLoss(visdom_logger, "D(G(X)) | X", every=1), Event.ON_EPOCH_END) \
                .with_event_handler(PlotCustomLoss(visdom_logger, "Discriminator Loss", every=1),
                                    Event.ON_EPOCH_END) \
                .with_event_handler(PlotCustomLoss(visdom_logger, "Total Loss", every=1), Event.ON_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger, "Jensen-Shannon Divergence", every=1,
                                             params={"title": "Jensen-Shannon Divergence on test data per Epoch",
                                                     "legend": ["Inputs", "Normalized"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger, "Per Dataset Mean Hausdorff Distance", every=1,
                                             params={"title": "Per Dataset Mean Hausdorff Distance",
                                                     "legend": list(dataset_configs.keys())}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger, "Dice score per class per epoch", every=1,
                                             params={"title": "Dice score on test patches per class per epoch",
                                                     "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger,
                                             "Hausdorff Distance per class per epoch on reconstructed iSEG image",
                                             every=1,
                                             params={
                                                 "title": "Hausdorff Distance score per class per epoch on reconstructed iSEG image",
                                                 "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger,
                                             "Hausdorff Distance per class per epoch on reconstructed MRBrainS image",
                                             every=1,
                                             params={
                                                 "title": "Hausdorff Distance per class per epoch on reconstructed MRBrainS image",
                                                 "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger,
                                             "Hausdorff Distance per class per epoch on reconstructed ABIDE image",
                                             every=1,
                                             params={
                                                 "title": "Hausdorff Distance per class per epoch on reconstructed ABIDE image",
                                                 "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger,
                                             "Dice score per class per epoch on reconstructed iSEG image",
                                             every=1,
                                             params={
                                                 "title": "Dice score per class per epoch on reconstructed iSEG image",
                                                 "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger,
                                             "Dice score per class per epoch on reconstructed MRBrainS image", every=1,
                                             params={
                                                 "title": "Dice score per class per epoch on reconstructed MRBrainS image",
                                                 "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger,
                                             "Dice score per class per epoch on reconstructed ABIDE image",
                                             every=1,
                                             params={
                                                 "title": "Dice score per class per epoch on reconstructed ABIDE image",
                                                 "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Input iSEG Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Input iSEG Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Normalized iSEG Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Normalized iSEG Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Segmented iSEG Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Segmented iSEG Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Ground Truth iSEG Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Ground Truth iSEG Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Initial Noise iSEG Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Initial Noise iSEG Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Noise iSEG After Normalization", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Noise iSEG After Normalization"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Input MRBrainS Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Input MRBrainS Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Normalized MRBrainS Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Normalized MRBrainS Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Segmented MRBrainS Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Segmented MRBrainS Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Ground Truth MRBrainS Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Ground Truth MRBrainS Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Initial Noise MRBrainS Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Initial Noise MRBrainS Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Noise MRBrainS After Normalization",
                                    PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Noise MRBrainS After Normalization"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Input ABIDE Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Input ABIDE Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Normalized ABIDE Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Normalized ABIDE Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Ground Truth ABIDE Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Ground Truth ABIDE Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Segmented ABIDE Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Segmented ABIDE Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Conv1 FM", PlotType.IMAGES_PLOT,
                                    params={"nrow": 8, "opts": {"store_history": True,
                                                                "title": "Conv1 FM"}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Layer1 FM", PlotType.IMAGES_PLOT,
                                    params={"nrow": 8, "opts": {"store_history": True,
                                                                "title": "Layer1 FM"}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Layer2 FM", PlotType.IMAGES_PLOT,
                                    params={"nrow": 12, "opts": {"store_history": True,
                                                                 "title": "Layer2 FM"}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Layer3 FM", PlotType.IMAGES_PLOT,
                                    params={"nrow": 16, "opts": {"store_history": True,
                                                                 "title": "Layer3 FM"}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Per-Dataset Histograms", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Images Histograms", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True}}, every=5), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                Checkpoint(save_folder, monitor_fn=lambda model_trainer: model_trainer.valid_loss, delta=0.01,
                           mode=MonitorMode.MIN), Event.ON_EPOCH_END) \
                .with_event_handler(PlotAvgGradientPerLayer(visdom_logger, every=25), Event.ON_TRAIN_BATCH_END)

            return trainer

        elif self._trainer == TrainerType.LSGAN:
            trainer = LSGANTrainer(training_config, model_trainers, dataloaders[0], dataloaders[1],
                                   dataloaders[2],
                                   reconstruction_datasets, normalized_reconstructors, input_reconstructors,
                                   segmentation_reconstructors, augmented_input_reconstructors,
                                   gt_reconstructors,
                                   run_config, dataset_configs, save_folder) \
                .with_event_handler(PrintTrainingStatus(every=25), Event.ON_BATCH_END) \
                .with_event_handler(PrintMonitors(every=25), Event.ON_BATCH_END) \
                .with_event_handler(PlotMonitors(visdom_logger), Event.ON_EPOCH_END) \
                .with_event_handler(PlotLR(visdom_logger), Event.ON_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Training Generated Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Training Generated Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Validation Generated Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Validation Generated Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_VALID_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Test Generated Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Test Generated Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Training Input Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Training Input Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Validation Input Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Validation Input Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_VALID_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Test Input Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Test Input Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Training Segmented Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Segmented Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Validation Segmented Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Validation Segmented Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_VALID_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Test Segmented Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Test Segmented Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Training Segmentation Ground Truth Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Training Ground Truth Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Validation Segmentation Ground Truth Batch Process {}".format(
                                        run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Validation Ground Truth Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_VALID_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Test Segmentation Ground Truth Batch Process {}".format(
                                        run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Test Ground Truth Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Training Label Map Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Training Label Map Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Validation Label Map Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Validation Label Map Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_VALID_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Test Label Map Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Test Label Map Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Generated Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "Generated Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Background Generated Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "Background Generated Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "CSF Generated Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "CSF Generated Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "GM Generated Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "GM Generated Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "WM Generated Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "WM Generated Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "Inputs Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Background Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "Background Input Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "CSF Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "CSF Input Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "GM Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "GM Input Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "WM Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "WM Input Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Pie Plot", PlotType.PIE_PLOT,
                                                        params={"opts": {"title": "Classification hit per classes",
                                                                         "legend": list(map(lambda key: key,
                                                                                            dataset_configs.keys())) + [
                                                                                       "Fake Class"]}},
                                                        every=25), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Pie Plot True", PlotType.PIE_PLOT,
                                                        params={"opts": {"title": "Batch data distribution",
                                                                         "legend": list(map(lambda key: key,
                                                                                            dataset_configs.keys())) + [
                                                                                       "Fake Class"]}},
                                                        every=25), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Mean Hausdorff Distance", PlotType.LINE_PLOT,
                                                        params={"opts": {"title": "Mean Hausdorff Distance",
                                                                         "legend": ["Test"]}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Metric Table", PlotType.TEXT_PLOT,
                                                        params={"opts": {"title": "Metric Table"}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Per-Dataset Metric Table", PlotType.TEXT_PLOT,
                                                        params={"opts": {"title": "Per-Dataset Metric Table"}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Jensen-Shannon Table", PlotType.TEXT_PLOT,
                                                        params={"opts": {"title": "Jensen-Shannon Divergence"}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Confusion Matrix", PlotType.HEATMAP_PLOT,
                                                        params={
                                                            "opts": {"columnnames": ["VM", "GM", "CSF", "Background"],
                                                                     "rownames": ["Background", "CSF", "GM", "WM"],
                                                                     "title": "Confusion Matrix"}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "iSEG Confusion Matrix", PlotType.HEATMAP_PLOT,
                                                        params={
                                                            "opts": {"columnnames": ["VM", "GM", "CSF", "Background"],
                                                                     "rownames": ["Background", "CSF", "GM", "WM"],
                                                                     "title": "iSEG Confusion Matrix"}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "MRBrainS Confusion Matrix", PlotType.HEATMAP_PLOT,
                                    params={"opts": {"columnnames": ["VM", "GM", "CSF", "Background"],
                                                     "rownames": ["Background", "CSF", "GM", "WM"],
                                                     "title": "MRBrainS Confusion Matrix"}},
                                    every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "ABIDE Confusion Matrix", PlotType.HEATMAP_PLOT,
                                                        params={
                                                            "opts": {"columnnames": ["VM", "GM", "CSF", "Background"],
                                                                     "rownames": ["Background", "CSF", "GM", "WM"],
                                                                     "title": "ABIDE Confusion Matrix"}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Discriminator Confusion Matrix", PlotType.HEATMAP_PLOT,
                                    params={"opts": {
                                        "columnnames": ["Generated"] + list(reversed(list(dataset_configs.keys()))),
                                        "rownames": list(dataset_configs.keys()) + ["Generated"],
                                        "title": "Discriminator Confusion Matrix"}},
                                    every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Discriminator Confusion Matrix Training", PlotType.HEATMAP_PLOT,
                                    params={"opts": {
                                        "columnnames": ["Generated"] + list(reversed(list(dataset_configs.keys()))),
                                        "rownames": list(dataset_configs.keys()) + ["Generated"],
                                        "title": "Discriminator Confusion Matrix Training"}},
                                    every=1), Event.ON_TRAIN_EPOCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Runtime", PlotType.TEXT_PLOT,
                                                        params={"opts": {"title": "Runtime"}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(PlotCustomLoss(visdom_logger, "D(G(X)) | X", every=1), Event.ON_EPOCH_END) \
                .with_event_handler(PlotCustomLoss(visdom_logger, "Discriminator Loss", every=1),
                                    Event.ON_EPOCH_END) \
                .with_event_handler(PlotCustomLoss(visdom_logger, "Total Loss", every=1), Event.ON_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger, "Jensen-Shannon Divergence", every=1,
                                             params={"title": "Jensen-Shannon Divergence on test data per Epoch",
                                                     "legend": ["Inputs", "Normalized"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger, "Dice score per class per epoch", every=1,
                                             params={"title": "Dice score on test patches per class per epoch",
                                                     "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger, "Per Dataset Mean Hausdorff Distance", every=1,
                                             params={"title": "Per Dataset Mean Hausdorff Distance",
                                                     "legend": list(dataset_configs.keys())}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger,
                                             "Hausdorff Distance per class per epoch on reconstructed iSEG image",
                                             every=1,
                                             params={
                                                 "title": "Hausdorff Distance score per class per epoch on reconstructed iSEG image",
                                                 "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger,
                                             "Hausdorff Distance per class per epoch on reconstructed MRBrainS image",
                                             every=1,
                                             params={
                                                 "title": "Hausdorff Distance per class per epoch on reconstructed MRBrainS image",
                                                 "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger,
                                             "Hausdorff Distance per class per epoch on reconstructed ABIDE image",
                                             every=1,
                                             params={
                                                 "title": "Hausdorff Distance per class per epoch on reconstructed ABIDE image",
                                                 "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger,
                                             "Dice score per class per epoch on reconstructed iSEG image",
                                             every=1,
                                             params={
                                                 "title": "Dice score per class per epoch on reconstructed iSEG image",
                                                 "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger,
                                             "Dice score per class per epoch on reconstructed MRBrainS image", every=1,
                                             params={
                                                 "title": "Dice score per class per epoch on reconstructed MRBrainS image",
                                                 "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger,
                                             "Dice score per class per epoch on reconstructed ABIDE image",
                                             every=1,
                                             params={
                                                 "title": "Dice score per class per epoch on reconstructed ABIDE image",
                                                 "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Input iSEG Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Input iSEG Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Normalized iSEG Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Normalized iSEG Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Segmented iSEG Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Segmented iSEG Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Ground Truth iSEG Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Ground Truth iSEG Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Initial Noise iSEG Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Initial Noise iSEG Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Noise iSEG After Normalization", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Noise iSEG After Normalization"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Input MRBrainS Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Input MRBrainS Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Normalized MRBrainS Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Normalized MRBrainS Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Segmented MRBrainS Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Segmented MRBrainS Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Ground Truth MRBrainS Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Ground Truth MRBrainS Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Initial Noise MRBrainS Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Initial Noise MRBrainS Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Noise MRBrainS After Normalization",
                                    PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Noise MRBrainS After Normalization"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Input ABIDE Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Input ABIDE Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Normalized ABIDE Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Normalized ABIDE Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Ground Truth ABIDE Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Ground Truth ABIDE Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Segmented ABIDE Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Segmented ABIDE Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Conv1 FM", PlotType.IMAGES_PLOT,
                                    params={"nrow": 8, "opts": {"store_history": True,
                                                                "title": "Conv1 FM"}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Layer1 FM", PlotType.IMAGES_PLOT,
                                    params={"nrow": 8, "opts": {"store_history": True,
                                                                "title": "Layer1 FM"}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Layer2 FM", PlotType.IMAGES_PLOT,
                                    params={"nrow": 12, "opts": {"store_history": True,
                                                                 "title": "Layer2 FM"}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Layer3 FM", PlotType.IMAGES_PLOT,
                                    params={"nrow": 16, "opts": {"store_history": True,
                                                                 "title": "Layer3 FM"}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Per-Dataset Histograms", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Images Histograms", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True}}, every=5), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                Checkpoint(save_folder, monitor_fn=lambda model_trainer: model_trainer.valid_loss, delta=0.01,
                           mode=MonitorMode.MIN), Event.ON_EPOCH_END) \
                .with_event_handler(PlotAvgGradientPerLayer(visdom_logger, every=25), Event.ON_TRAIN_BATCH_END)

            return trainer

        elif self._trainer == TrainerType.DUALUNET:
            trainer = DualUNetTrainer(training_config, model_trainers, dataloaders[0], dataloaders[1], dataloaders[2],
                                      reconstruction_datasets, normalized_reconstructors, input_reconstructors,
                                      segmentation_reconstructors, augmented_input_reconstructors, gt_reconstructors,
                                      run_config, dataset_configs, save_folder) \
                .with_event_handler(PrintTrainingStatus(every=25), Event.ON_BATCH_END) \
                .with_event_handler(PrintMonitors(every=25), Event.ON_BATCH_END) \
                .with_event_handler(PlotMonitors(visdom_logger), Event.ON_EPOCH_END) \
                .with_event_handler(PlotLR(visdom_logger), Event.ON_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Training Generated Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Training Generated Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Validation Generated Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Validation Generated Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_VALID_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Test Generated Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Test Generated Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Training Input Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Training Input Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Validation Input Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Validation Input Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_VALID_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Test Input Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Test Input Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Training Segmented Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Segmented Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Validation Segmented Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Validation Segmented Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_VALID_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Test Segmented Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Test Segmented Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Training Segmentation Ground Truth Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Training Ground Truth Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Validation Segmentation Ground Truth Batch Process {}".format(
                                        run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Validation Ground Truth Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_VALID_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Test Segmentation Ground Truth Batch Process {}".format(
                                        run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Test Ground Truth Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Training Label Map Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Training Label Map Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Validation Label Map Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Validation Label Map Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_VALID_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Test Label Map Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Test Label Map Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Generated Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "Generated Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Background Generated Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "Background Generated Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "CSF Generated Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "CSF Generated Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "GM Generated Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "GM Generated Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "WM Generated Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "WM Generated Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={
                                        "opts": {"title": "Inputs Intensity Histogram",
                                                 "store_history": True,
                                                 "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Background Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "Background Input Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "CSF Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "CSF Input Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "GM Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "GM Input Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "WM Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "WM Input Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Mean Hausdorff Distance", PlotType.LINE_PLOT,
                                                        params={"opts": {"title": "Mean Hausdorff Distance",
                                                                         "legend": ["Test"]}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Metric Table", PlotType.TEXT_PLOT,
                                                        params={"opts": {"title": "Metric Table"}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Per-Dataset Metric Table", PlotType.TEXT_PLOT,
                                                        params={"opts": {"title": "Per-Dataset Metric Table"}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Jensen-Shannon Table", PlotType.TEXT_PLOT,
                                                        params={"opts": {"title": "Jensen-Shannon Divergence"}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Confusion Matrix", PlotType.HEATMAP_PLOT,
                                                        params={
                                                            "opts": {"columnnames": ["VM", "GM", "CSF", "Background"],
                                                                     "rownames": ["Background", "CSF", "GM", "WM"],
                                                                     "title": "Confusion Matrix"}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "iSEG Confusion Matrix", PlotType.HEATMAP_PLOT,
                                                        params={
                                                            "opts": {"columnnames": ["VM", "GM", "CSF", "Background"],
                                                                     "rownames": ["Background", "CSF", "GM", "WM"],
                                                                     "title": "iSEG Confusion Matrix"}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "MRBrainS Confusion Matrix", PlotType.HEATMAP_PLOT,
                                    params={"opts": {"columnnames": ["VM", "GM", "CSF", "Background"],
                                                     "rownames": ["Background", "CSF", "GM", "WM"],
                                                     "title": "MRBrainS Confusion Matrix"}},
                                    every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "ABIDE Confusion Matrix", PlotType.HEATMAP_PLOT,
                                                        params={
                                                            "opts": {"columnnames": ["VM", "GM", "CSF", "Background"],
                                                                     "rownames": ["Background", "CSF", "GM", "WM"],
                                                                     "title": "ABIDE Confusion Matrix"}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Runtime", PlotType.TEXT_PLOT,
                                                        params={"opts": {"title": "Runtime"}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger, "Jensen-Shannon Divergence", every=1,
                                             params={"title": "Jensen-Shannon Divergence on test data per Epoch",
                                                     "legend": ["Inputs", "Normalized"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger,
                                             "Hausdorff Distance per class per epoch on reconstructed iSEG image",
                                             every=1,
                                             params={
                                                 "title": "Hausdorff Distance score per class per epoch on reconstructed iSEG image",
                                                 "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger,
                                             "Hausdorff Distance per class per epoch on reconstructed MRBrainS image",
                                             every=1,
                                             params={
                                                 "title": "Hausdorff Distance per class per epoch on reconstructed MRBrainS image",
                                                 "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger,
                                             "Hausdorff Distance per class per epoch on reconstructed ABIDE image",
                                             every=1,
                                             params={
                                                 "title": "Hausdorff Distance per class per epoch on reconstructed ABIDE image",
                                                 "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger, "Dice score per class per epoch", every=1,
                                             params={"title": "Dice score on test patches per class per epoch",
                                                     "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger,
                                             "Dice score per class per epoch on reconstructed iSEG image",
                                             every=1,
                                             params={
                                                 "title": "Dice score per class per epoch on reconstructed iSEG image",
                                                 "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger,
                                             "Dice score per class per epoch on reconstructed MRBrainS image", every=1,
                                             params={
                                                 "title": "Dice score per class per epoch on reconstructed MRBrainS image",
                                                 "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger,
                                             "Dice score per class per epoch on reconstructed ABIDE image",
                                             every=1,
                                             params={
                                                 "title": "Dice score per class per epoch on reconstructed ABIDE image",
                                                 "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Input iSEG Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Input iSEG Image"}},
                                    every=100), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Normalized iSEG Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Normalized iSEG Image"}},
                                    every=100), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Segmented iSEG Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Segmented iSEG Image"}},
                                    every=100), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Ground Truth iSEG Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Ground Truth iSEG Image"}},
                                    every=100), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Initial Noise iSEG Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Initial Noise iSEG Image"}},
                                    every=100), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Noise iSEG After Normalization", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Noise iSEG After Normalization"}},
                                    every=100), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Input MRBrainS Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Input MRBrainS Image"}},
                                    every=100), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Normalized MRBrainS Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Normalized MRBrainS Image"}},
                                    every=100), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Segmented MRBrainS Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Segmented MRBrainS Image"}},
                                    every=100), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Ground Truth MRBrainS Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Ground Truth MRBrainS Image"}},
                                    every=100), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Initial Noise MRBrainS Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Initial Noise MRBrainS Image"}},
                                    every=100), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Noise MRBrainS After Normalization",
                                    PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Noise MRBrainS After Normalization"}},
                                    every=100), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Input ABIDE Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Input ABIDE Image"}},
                                    every=100), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Normalized ABIDE Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Normalized ABIDE Image"}},
                                    every=100), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Ground Truth ABIDE Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Ground Truth ABIDE Image"}},
                                    every=100), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Segmented ABIDE Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Segmented ABIDE Image"}},
                                    every=100), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger, "Per Dataset Mean Hausdorff Distance", every=1,
                                             params={"title": "Per Dataset Mean Hausdorff Distance",
                                                     "legend": list(dataset_configs.keys())}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Per-Dataset Histograms", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Images Histograms", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True}}, every=5), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                Checkpoint(save_folder, monitor_fn=lambda model_trainer: model_trainer.valid_loss, delta=0.01,
                           mode=MonitorMode.MIN), Event.ON_EPOCH_END) \
                .with_event_handler(PlotAvgGradientPerLayer(visdom_logger, every=25), Event.ON_TRAIN_BATCH_END)
            return trainer

        elif self._trainer == TrainerType.UNET:
            trainer = UNetTrainer(training_config, model_trainers, dataloaders[0], dataloaders[1], dataloaders[2],
                                  reconstruction_datasets, input_reconstructors,
                                  segmentation_reconstructors, augmented_input_reconstructors, gt_reconstructors,
                                  run_config, dataset_configs, save_folder) \
                .with_event_handler(PrintTrainingStatus(every=25), Event.ON_BATCH_END) \
                .with_event_handler(PrintMonitors(every=25), Event.ON_BATCH_END) \
                .with_event_handler(PlotMonitors(visdom_logger), Event.ON_EPOCH_END) \
                .with_event_handler(PlotLR(visdom_logger), Event.ON_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Training Input Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Training Input Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Validation Input Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Validation Input Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_VALID_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Test Input Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Test Input Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Training Segmented Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Segmented Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Validation Segmented Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Validation Segmented Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_VALID_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Test Segmented Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Test Segmented Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Training Segmentation Ground Truth Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Training Ground Truth Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Validation Segmentation Ground Truth Batch Process {}".format(
                                        run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Validation Ground Truth Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_VALID_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Test Segmentation Ground Truth Batch Process {}".format(
                                        run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Test Ground Truth Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Training Label Map Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Training Label Map Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=500), Event.ON_TRAIN_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Validation Label Map Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Validation Label Map Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_VALID_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger,
                                    "Test Label Map Batch Process {}".format(run_config.local_rank),
                                    PlotType.IMAGES_PLOT,
                                    params={"nrow": 4,
                                            "opts": {"store_history": True,
                                                     "title": "Test Label Map Patches Process {}".format(
                                                         run_config.local_rank)}},
                                    every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={
                                        "opts": {"title": "Inputs Intensity Histogram",
                                                 "store_history": True,
                                                 "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Background Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "Background Input Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "CSF Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "CSF Input Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "GM Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "GM Input Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "WM Input Intensity Histogram", PlotType.HISTOGRAM_PLOT,
                                    params={"opts": {"title": "WM Input Intensity Histogram",
                                                     "store_history": True,
                                                     "numbins": 128}}, every=100), Event.ON_TEST_BATCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Mean Hausdorff Distance", PlotType.LINE_PLOT,
                                                        params={"opts": {"title": "Mean Hausdorff Distance",
                                                                         "legend": ["Test"]}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Metric Table", PlotType.TEXT_PLOT,
                                                        params={"opts": {"title": "Metric Table"}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Per-Dataset Metric Table", PlotType.TEXT_PLOT,
                                                        params={"opts": {"title": "Per-Dataset Metric Table"}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Confusion Matrix", PlotType.HEATMAP_PLOT,
                                                        params={
                                                            "opts": {"columnnames": ["VM", "GM", "CSF", "Background"],
                                                                     "rownames": ["Background", "CSF", "GM", "WM"],
                                                                     "title": "Confusion Matrix"}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "iSEG Confusion Matrix", PlotType.HEATMAP_PLOT,
                                                        params={
                                                            "opts": {"columnnames": ["VM", "GM", "CSF", "Background"],
                                                                     "rownames": ["Background", "CSF", "GM", "WM"],
                                                                     "title": "iSEG Confusion Matrix"}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "MRBrainS Confusion Matrix", PlotType.HEATMAP_PLOT,
                                    params={"opts": {"columnnames": ["VM", "GM", "CSF", "Background"],
                                                     "rownames": ["Background", "CSF", "GM", "WM"],
                                                     "title": "MRBrainS Confusion Matrix"}},
                                    every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "ABIDE Confusion Matrix", PlotType.HEATMAP_PLOT,
                                                        params={
                                                            "opts": {"columnnames": ["VM", "GM", "CSF", "Background"],
                                                                     "rownames": ["Background", "CSF", "GM", "WM"],
                                                                     "title": "ABIDE Confusion Matrix"}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(PlotCustomVariables(visdom_logger, "Runtime", PlotType.TEXT_PLOT,
                                                        params={"opts": {"title": "Runtime"}},
                                                        every=1), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger, "Dice score per class per epoch", every=1,
                                             params={"title": "Dice score on test patches per class per epoch",
                                                     "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger,
                                             "Hausdorff Distance per class per epoch on reconstructed iSEG image",
                                             every=1,
                                             params={
                                                 "title": "Hausdorff Distance score per class per epoch on reconstructed iSEG image",
                                                 "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger,
                                             "Hausdorff Distance per class per epoch on reconstructed MRBrainS image",
                                             every=1,
                                             params={
                                                 "title": "Hausdorff Distance per class per epoch on reconstructed MRBrainS image",
                                                 "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger,
                                             "Hausdorff Distance per class per epoch on reconstructed ABIDE image",
                                             every=1,
                                             params={
                                                 "title": "Hausdorff Distance per class per epoch on reconstructed ABIDE image",
                                                 "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger,
                                             "Dice score per class per epoch on reconstructed iSEG image",
                                             every=1,
                                             params={
                                                 "title": "Dice score per class per epoch on reconstructed iSEG image",
                                                 "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger,
                                             "Dice score per class per epoch on reconstructed MRBrainS image", every=1,
                                             params={
                                                 "title": "Dice score per class per epoch on reconstructed MRBrainS image",
                                                 "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger,
                                             "Dice score per class per epoch on reconstructed ABIDE image",
                                             every=1,
                                             params={
                                                 "title": "Dice score per class per epoch on reconstructed ABIDE image",
                                                 "legend": ["CSF", "GM", "WM"]}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Input iSEG Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Input iSEG Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Segmented iSEG Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Segmented iSEG Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Ground Truth iSEG Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Ground Truth iSEG Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Input MRBrainS Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Input MRBrainS Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Segmented MRBrainS Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Segmented MRBrainS Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Ground Truth MRBrainS Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Ground Truth MRBrainS Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Input ABIDE Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Input ABIDE Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Ground Truth ABIDE Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Ground Truth ABIDE Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomVariables(visdom_logger, "Reconstructed Segmented ABIDE Image", PlotType.IMAGE_PLOT,
                                    params={"opts": {"store_history": True,
                                                     "title": "Reconstructed Segmented ABIDE Image"}},
                                    every=10), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                PlotCustomLinePlotWithLegend(visdom_logger, "Per Dataset Mean Hausdorff Distance", every=1,
                                             params={"title": "Per Dataset Mean Hausdorff Distance",
                                                     "legend": list(dataset_configs.keys())}), Event.ON_TEST_EPOCH_END) \
                .with_event_handler(
                Checkpoint(save_folder, monitor_fn=lambda model_trainer: model_trainer.valid_loss, delta=0.01,
                           mode=MonitorMode.MIN), Event.ON_EPOCH_END) \
                .with_event_handler(PlotAvgGradientPerLayer(visdom_logger, every=25), Event.ON_TRAIN_BATCH_END)
            return trainer
