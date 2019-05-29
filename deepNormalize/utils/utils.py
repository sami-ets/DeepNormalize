import re
import os


def get_MRBrainS_subjects(path):
    """
    Utility method to get, filter and arrange BraTS data set in a series of lists.
    Args:
        path: The path to look for BraTS data set files.

    Returns:
        A tuple containing multimodal MRI images for each subject and their respective segmentation.
    """
    subjects = list()
    keys = ["t1", "t1_1mm", "t1_ir", "t2", "roi", "LabelsForTraining", "LabelsForTesting"]

    for (dirpath, dirnames, filenames) in os.walk(path):
        if len(filenames) is not 0:
            # Filter files.
            t1 = list(filter(re.compile(r"^T1.nii").search, filenames))
            t1_1mm = list(filter(re.compile(r"^T1_1mm.nii").search, filenames))
            t1_ir = list(filter(re.compile(r"^T1_IR.nii").search, filenames))
            t2 = list(filter(re.compile(r"^T2_FLAIR.nii").search, filenames))
            roi = list(filter(re.compile(r"^ROIT1.nii").search, filenames))
            seg_training = list(filter(re.compile(r"^LabelsForTraining.nii").search, filenames))
            seg_testing = list(filter(re.compile(r"^LabelsForTesting.nii").search, filenames))

            t1 = [os.path.join(dirpath, ("{}".format(i))) for i in t1]
            t1_1mm = [os.path.join(dirpath, ("{}".format(i))) for i in t1_1mm]
            t1_ir = [os.path.join(dirpath, ("{}".format(i))) for i in t1_ir]
            t2 = [os.path.join(dirpath, ("{}".format(i))) for i in t2]
            roi = [os.path.join(dirpath, ("{}".format(i))) for i in roi]
            seg_training = [os.path.join(dirpath, ("{}".format(i))) for i in seg_training]
            seg_testing = [os.path.join(dirpath, ("{}".format(i))) for i in seg_testing]

            subjects.append(dict((key, volume) for key, volume in zip(keys, [t1,
                                                                             t1_1mm,
                                                                             t1_ir,
                                                                             t2,
                                                                             roi,
                                                                             seg_training,
                                                                             seg_testing])))

    return subjects

def get_iSEG_subjects(path):
    """
        Utility method to get, filter and arrange BraTS data set in a series of lists.
        Args:
            path: The path to look for BraTS data set files.

        Returns:
            A tuple containing multimodal MRI images for each subject and their respective segmentation.
        """

    subjects = list()
    keys = ["t1", "t2", "roi", "label"]

    for (dirpath, dirnames, filenames) in os.walk(path):
        if len(filenames) is not 0:
            # Filter files.
            t1 = list(filter(re.compile(r"^.*?T1.nii$").search, filenames))
            t2 = list(filter(re.compile(r"^.*?T2.nii$").search, filenames))
            roi = list(filter(re.compile(r"^.*?ROIT1.nii.gz$").search, filenames))
            seg_training = list(filter(re.compile(r"^.*?labels.nii$").search, filenames))

            t1 = [os.path.join(dirpath, ("{}".format(i))) for i in t1]
            t2 = [os.path.join(dirpath, ("{}".format(i))) for i in t2]
            roi = [os.path.join(dirpath, ("{}".format(i))) for i in roi]
            seg_training = [os.path.join(dirpath, ("{}".format(i))) for i in seg_training]

            subjects.append(dict((key, volume) for key, volume in zip(keys, [t1,
                                                                             t2,
                                                                             roi,
                                                                             seg_training])))

    return subjects