import paramiko
import pandas
import os

BASE_PATH = "/project/def-lombaert/pld2602/"

if __name__ == "__main__":

    iseg_csv = pandas.read_csv("/data/users/pldelisle/datasets/Preprocessed_4/iSEG/Training/output.csv")
    mrbrains_csv = pandas.read_csv(
        "/data/users/pldelisle/datasets/Preprocessed_4/MRBrainS/DataNii/TrainingData/output.csv")

    filtered_iSEG_csv = iseg_csv.loc[iseg_csv["center_class"].isin([1, 2, 3])]
    filtered_MRBrainS_csv = mrbrains_csv.loc[mrbrains_csv["center_class"].isin([1, 2, 3])]

    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname="beluga.computecanada.ca", username="pld2602", password="TeslaV100!")

    sftp_client = ssh_client.open_sftp()

    for index, row in filtered_iSEG_csv.iterrows():
        T_1_relpath = os.path.relpath(row["T1"], "/data/users/pldelisle/datasets/")
        T_2_relpath = os.path.relpath(row["T2"], "/data/users/pldelisle/datasets/")
        label_relpath = os.path.relpath(row["labels"], "/data/users/pldelisle/datasets/")

        sftp_client.put(row["T1"], BASE_PATH + T_1_relpath)
        sftp_client.put(row["T2"], BASE_PATH + T_2_relpath)
        sftp_client.put(row["labels"], BASE_PATH + label_relpath)

    for index, row in filtered_MRBrainS_csv.iterrows():
        T_1_relpath = os.path.relpath(row["T1"], "/data/users/pldelisle/datasets/")
        T_2_relpath = os.path.relpath(row["T2_FLAIR"], "/data/users/pldelisle/datasets/")
        label_relpath = os.path.relpath(row["LabelsForTesting"], "/data/users/pldelisle/datasets/")
        sftp_client.put(row["T1"], "/project/def-lombaert/pld2602/" + T_1_relpath)
        sftp_client.put(row["T2_FLAIR"], "/project/def-lombaert/pld2602/" + T_2_relpath)
        sftp_client.put(row["LabelsForTesting"], "/project/def-lombaert/pld2602/" + label_relpath)

    sftp_client.close()
    ssh_client.close()
