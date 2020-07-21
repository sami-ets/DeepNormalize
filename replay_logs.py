import multiprocessing

import math
import pandas
import visdom

# find $directory -type f -name "*.in"

class LogParser(object):

    def __init__(self, file_path):
        self._file_path = file_path

    @staticmethod
    def chunks(l, n):
        return [l[i:i + n] for i in range(0, len(l), n)]

    def _do_job(self, paths):
        vis = visdom.Visdom()

        for path in paths:
            vis.replay_log(path)

    def _dispatch_jobs(self, files, job_number):
        total = len(files)
        chunk_size = math.ceil(total / job_number)
        slices = self.chunks(files, chunk_size)
        jobs = []

        for slice in slices:
            j = multiprocessing.Process(target=self._do_job, args=(slice,))
            jobs.append(j)
        for j in jobs:
            j.start()

    def run(self, **kwargs):
        paths = pandas.read_csv(self._file_path)["paths"]

        self._dispatch_jobs(paths, 8)


if __name__ == '__main__':
    log_parser = LogParser("json_paths.csv").run()
