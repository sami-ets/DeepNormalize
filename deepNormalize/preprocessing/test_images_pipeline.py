import argparse

from samitorch.inputs.transformers import ToNumpyArray
from torchvision.transforms import transforms

from deepNormalize.preprocessing.pipelines import iSEGPreProcessingPipeline, MRBrainsPreProcessingPipeline, \
    AnatomicalPreProcessingPipeline, AlignPipeline, MRBrainSPatchPreProcessingPipeline, FlipLR, Transpose, \
    iSEGPatchPreProcessingPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-iseg', type=str, help='Path to the iSEG preprocessed directory.', required=True)
    parser.add_argument('--path-mrbrains', type=str, help='Path to the preprocessed directory.', required=True)

    args = parser.parse_args()
    # iSEGPreProcessingPipeline(root_dir=args.path_iseg,
    #                           output_dir="/mnt/md0/Data/Preprocessed/iSEG/TestingData/Preprocessed").run()
    # MRBrainsPreProcessingPipeline(root_dir=args.path_mrbrains,
    #                               output_dir="/mnt/md0/Data/Preprocessed/MRBrainS/TestData/Preprocessed").run()

    # AnatomicalPreProcessingPipeline(root_dir="/mnt/md0/Data/Preprocessed/MRBrainS/TestData/Preprocessed",
    #                                 output_dir="/mnt/md0/Data/Preprocessed/MRBrainS/TestDataSizeNormalized").run()
    AnatomicalPreProcessingPipeline(root_dir="/mnt/md0/Data/Preprocessed/iSEG/TestingData/Preprocessed",
                                    output_dir="/mnt/md0/Data/Preprocessed/iSEG/TestingData/SizeNormalized").run()

    AlignPipeline(root_dir="/mnt/md0/Data/Preprocessed/iSEG/TestingData/SizeNormalized",
                  transforms=transforms.Compose([ToNumpyArray(),
                                                 FlipLR()]),
                  output_dir="/mnt/md0/Data/Preprocessed/iSEG/TestingData/Aligned"
                  ).run()
    # AlignPipeline(root_dir="/mnt/md0/Data/Preprocessed/MRBrainS/TestData/SizeNormalized",
    #               transforms=transforms.Compose([ToNumpyArray(),
    #                                              Transpose((0, 2, 3, 1))]),
    #               output_dir="/mnt/md0/Data/Preprocessed/MRBrainS/TestData/Aligned"
    #               ).run()

    # MRBrainSPatchPreProcessingPipeline(root_dir="/mnt/md0/Data/Preprocessed/MRBrainS/TestData/Aligned",
    #                                    output_dir="/mnt/md0/Data/Preprocessed/MRBrainS/TestData/Patches/Aligned",
    #                                    patch_size=(1, 32, 32, 32), step=(1, 8, 8, 8)).run(keep_forground_only=False)
    iSEGPatchPreProcessingPipeline(root_dir="/mnt/md0/Data/Preprocessed/iSEG/TestingData/Aligned",
                                   output_dir="/mnt/md0/Data/Preprocessed/iSEG/TestingData/Patches/Aligned",
                                   patch_size=(1, 32, 32, 32), step=(1, 8, 8, 8)).run(keep_foreground_only=False,
                                                                                      keep_labels=False)
