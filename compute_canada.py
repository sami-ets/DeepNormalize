import re

import paramiko
import pandas
import os
import logging
logging.basicConfig(level=logging.INFO)

BASE_PATH = "/project/def-lombaert/pld2602/"

if __name__ == "__main__":
    LOGGER = logging.getLogger("ABIDETransfer")

    # iseg_csv = pandas.read_csv("/data/users/pldelisle/datasets/Preprocessed_4/iSEG/Training/output.csv")
    # mrbrains_csv = pandas.read_csv(
    #     "/data/users/pldelisle/datasets/Preprocessed_4/MRBrainS/DataNii/TrainingData/output.csv")
    abide_csv = pandas.read_csv("/mnt/md0/Data/output_abide_images.csv")

    # filtered_iSEG_csv = iseg_csv.loc[iseg_csv["center_class"].isin([1, 2, 3])]
    # filtered_MRBrainS_csv = mrbrains_csv.loc[mrbrains_csv["center_class"].isin([1, 2, 3])]

    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname="beluga.computecanada.ca", username="pld2602", password="TeslaV100!")

    sftp_client = ssh_client.open_sftp()

    # for index, row in abide_csv.iterrows():
    #     sftp_client.mkdir(os.path.join(BASE_PATH, "ABIDE", "5.1", row["subjects"]))

    for index, row in abide_csv.iterrows():
        LOGGER.info("Processing subject {}".format(row["subjects"]))
        sftp_client.mkdir(os.path.join(BASE_PATH, "ABIDE", "5.1", row["subjects"], "mri"))
        sftp_client.put(row["T1"],
                        os.path.join(BASE_PATH, "ABIDE", "5.1", row["subjects"], "mri", "real_brainmask.nii.gz"))
        sftp_client.put(row["labels"],
                        os.path.join(BASE_PATH, "ABIDE", "5.1", row["subjects"], "mri", "aligned_labels.nii.gz"))

    sftp_client.close()
    ssh_client.close()
