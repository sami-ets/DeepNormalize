import unittest

import matplotlib.pyplot as plt
import torch
from hamcrest import *
from samitorch.inputs.images import Modality
from torch.utils.data.dataset import Dataset

from deepNormalize.inputs.datasets import iSEGSegmentationFactory, MRBrainSSegmentationFactory, ABIDESegmentationFactory


class iSEGSegmentationFactoryTest(unittest.TestCase):
    DATA_PATH = "/mnt/md0/Data/Preprocessed/iSEG/Training/Patches/Aligned"

    def setUp(self) -> None:
        pass

    def test_should_create_single_modality_train_valid_test_datasets(self):
        train_dataset, valid_dataset, test_dataset, reconstruction_dataset, csv = iSEGSegmentationFactory.create_train_valid_test(
            self.DATA_PATH + "/Filtered", self.DATA_PATH + "/Filtered/label", Modality.T1, 0, 0.2,
            self.DATA_PATH + "/Full")

        assert_that(train_dataset, instance_of(Dataset))
        assert_that(valid_dataset, instance_of(Dataset))
        assert_that(test_dataset, instance_of(Dataset))
        assert_that(reconstruction_dataset, instance_of(Dataset))

    def test_should_produce_a_single_modality_input_with_one_channel(self):
        train_dataset, valid_dataset, test_dataset, reconstruction_dataset, csv = iSEGSegmentationFactory.create_train_valid_test(
            self.DATA_PATH + "/Filtered", self.DATA_PATH + "/Filtered/label", Modality.T1, 0, 0.2,
            self.DATA_PATH + "/Full")

        sample = train_dataset[0]

        assert_that(sample.x.size(), is_(torch.Size([1, 32, 32, 32])))
        plt.imshow(sample.x[0, 16, :, :], cmap="gray")
        plt.show()
        plt.imshow(sample.y[0, 16, :, :], cmap="gray")
        plt.show()

    def test_should_create_single_modality_train_test_datasets(self):
        train_dataset, valid_dataset, reconstruction_dataset, csv = iSEGSegmentationFactory.create_train_test(
            self.DATA_PATH+ "/Filtered", self.DATA_PATH + "/Filtered/label", Modality.T1, 0, 0.2, self.DATA_PATH + "/Full")

        assert_that(train_dataset, instance_of(Dataset))
        assert_that(valid_dataset, instance_of(Dataset))
        assert_that(reconstruction_dataset, instance_of(Dataset))

    def test_should_produce_a_single_modality_input_with_one_channel_train_test(self):
        train_dataset, valid_dataset, reconstruction_dataset, csv = iSEGSegmentationFactory.create_train_test(
            self.DATA_PATH+ "/Filtered", self.DATA_PATH + "/Filtered/label", Modality.T1, 0, 0.2, self.DATA_PATH + "/Full")

        sample = train_dataset[0]

        assert_that(sample.x.size(), is_(torch.Size([1, 32, 32, 32])))
        plt.imshow(sample.x[0, 16, :, :], cmap="gray")
        plt.show()
        plt.imshow(sample.y[0, 16, :, :], cmap="gray")
        plt.show()

    def test_should_create_multimodal_train_valid_test_datasets(self):
        train_dataset, test_dataset, reconstruction_dataset, csv = iSEGSegmentationFactory.create_train_valid_test(
            self.DATA_PATH + "/Filtered", self.DATA_PATH + "/Filtered/label", [Modality.T1, Modality.T2], 0, 0.2, )

        assert_that(train_dataset, instance_of(Dataset))
        assert_that(test_dataset, instance_of(Dataset))
        assert_that(reconstruction_dataset, instance_of(Dataset))

    def test_should_produce_a_multimodal_input_with_two_channels(self):
        train_dataset, valid_dataset, test_dataset, reconstruction_dataset, csv = iSEGSegmentationFactory.create_train_valid_test(
            self.DATA_PATH, self.DATA_PATH + "/label", [Modality.T1, Modality.T2], 0, 0.2)

        sample = train_dataset[0]

        assert_that(sample.x.size(), is_(torch.Size([2, 32, 32, 32])))
        plt.imshow(sample.x[0, 16, :, :], cmap="gray")
        plt.show()
        plt.imshow(sample.x[1, 16, :, :], cmap="gray")
        plt.show()
        plt.imshow(sample.y[0, 16, :, :], cmap="gray")
        plt.show()


class MRBrainSSegmentationFactoryTest(unittest.TestCase):
    DATA_PATH = "/mnt/md0/Data/Preprocessed/MRBrainS/TrainingData/Patches/Aligned"

    def setUp(self) -> None:
        pass

    def test_should_create_single_modality_train_valid_test_datasets(self):
        train_dataset, valid_dataset, test_dataset, csv = MRBrainSSegmentationFactory.create_train_valid_test(
            self.DATA_PATH, self.DATA_PATH, Modality.T1, 0, 0.2)

        assert_that(train_dataset, instance_of(Dataset))
        assert_that(valid_dataset, instance_of(Dataset))
        assert_that(test_dataset, instance_of(Dataset))

    def test_should_produce_a_single_modality_input_with_one_channel(self):
        train_dataset, valid_dataset, test_dataset, csv = MRBrainSSegmentationFactory.create_train_valid_test(
            self.DATA_PATH, self.DATA_PATH, Modality.T1, 0, 0.2)

        sample = train_dataset[0]

        assert_that(sample.x.size(), is_(torch.Size([1, 32, 32, 32])))
        plt.imshow(sample.x[0, 16, :, :], cmap="gray")
        plt.show()
        plt.imshow(sample.y[0, 16, :, :], cmap="gray")
        plt.show()

    def test_should_create_multimodal_train_valid_test_datasets(self):
        train_dataset, valid_dataset, test_dataset, csv = MRBrainSSegmentationFactory.create_train_valid_test(
            self.DATA_PATH, self.DATA_PATH, [Modality.T1, Modality.T2_FLAIR], 0, 0.2)

        assert_that(train_dataset, instance_of(Dataset))
        assert_that(valid_dataset, instance_of(Dataset))
        assert_that(test_dataset, instance_of(Dataset))

    def test_should_produce_a_multimodal_input_with_two_channels(self):
        train_dataset, valid_dataset, test_dataset, csv = MRBrainSSegmentationFactory.create_train_valid_test(
            self.DATA_PATH, self.DATA_PATH, [Modality.T1, Modality.T2_FLAIR], 0, 0.2)

        sample = train_dataset[0]

        assert_that(sample.x.size(), is_(torch.Size([2, 32, 32, 32])))
        plt.imshow(sample.x[0, 16, :, :], cmap="gray")
        plt.show()
        plt.imshow(sample.x[1, 16, :, :], cmap="gray")
        plt.show()
        plt.imshow(sample.y[0, 16, :, :], cmap="gray")
        plt.show()


class ABIDESSegmentationFactoryTest(unittest.TestCase):
    DATA_PATH = "/home/pierre-luc-delisle/ABIDE/5.1/"

    def setUp(self) -> None:
        pass

    def test_should_create_single_modality_train_valid_test_datasets(self):
        train_dataset, valid_dataset, test_dataset, csv = ABIDESegmentationFactory.create_train_valid_test(
            self.DATA_PATH, self.DATA_PATH, Modality.T1, 0, 0.2)

        assert_that(train_dataset, instance_of(Dataset))
        assert_that(valid_dataset, instance_of(Dataset))
        assert_that(test_dataset, instance_of(Dataset))

    def test_should_produce_a_single_modality_input_with_one_channel(self):
        train_dataset, valid_dataset, test_dataset, csv = ABIDESegmentationFactory.create_train_valid_test(
            self.DATA_PATH, self.DATA_PATH, Modality.T1, 0, 0.2)

        sample = train_dataset[0]

        assert_that(sample.x.size(), is_(torch.Size([1, 32, 32, 32])))
        plt.imshow(sample.x[0, 16, :, :], cmap="gray")
        plt.show()
        plt.imshow(sample.y[0, 16, :, :], cmap="gray")
        plt.show()
