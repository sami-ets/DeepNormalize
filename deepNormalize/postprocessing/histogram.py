import pandas

if __name__ == '__main__':

    csv = pandas.read_csv("/mnt/md0/Research/Code/deepNormalizev5/histogram.csv")

    new_bins = list()
    new_count = list()

    for i in range(0, 127, 2):
        new_bins.append((csv.bins[i] + csv.bins[i + 1]) / 2)
    for i in range(0, 127, 2):
        new_count.append(csv.counts[i] + csv.counts[i + 1])

    headers = ["bins", "counts"]
    df = pandas.DataFrame([new_bins, new_count])

    df.to_csv("/mnt/md0/Research/Code/deepNormalizev5/new_histogram.csv")