# coding=utf-8
import logging

import dispy
import numpy as np
import pandas as pd
from sklearn.datasets.base import Bunch
from sklearn.preprocessing import LabelEncoder

from quantification.utils.errors import ClusterException


class ClusterParallel:
    def __init__(self, compute, iter_params, constant_named_params=None, dependencies=None, local=False, verbose=False):
        self.compute = compute
        self.iter_params = iter_params
        self.constant_params = constant_named_params
        if self.constant_params is None:
            self.constant_params = {}
        self.dependencies = dependencies
        if self.dependencies is None:
            self.dependencies = []
        self.local = local
        self.verbose = verbose

    def retrieve(self):

        results = []

        if self.local:
            for sample in self.iter_params:
                if isinstance(sample, tuple):
                    results.append(self.compute(*sample, **self.constant_params))
                else:
                    results.append(self.compute(sample, **self.constant_params))
            return np.array(results)

        cluster = dispy.SharedJobCluster(self.compute, depends=self.dependencies, loglevel=logging.WARNING,
                                         scheduler_node='dhcp015.aic.uniovi.es')

        jobs = []
        id = 0
        for sample in self.iter_params:
            if isinstance(sample, tuple):
                job = cluster.submit(*sample, **self.constant_params)
                job.id = id
                id += 1
            else:
                job = cluster.submit(sample, **self.constant_params)
                job.id = id
                id += 1
            jobs.append(job)
        # cluster.wait()

        for job in jobs:
            job()
            if job.exception is not None:
                raise ClusterException(job.exception + job.ip_addr)
            results.append(job.result)
        if self.verbose:
            cluster.print_status()
        cluster.close()
        return np.array(results)




if __name__ == '__main__':

    def compute(n, path):
        import time, socket, pickle
        time.sleep(n)
        host = socket.gethostname()
        with open(path, 'rb') as fh:
            bunch = pickle.load(fh)
            return (host, n, len(bunch))


    def load_plankton_file(path, sample_col="Sample", target_col="class"):

        data_file = pd.read_csv(path, delimiter=' ')
        le = LabelEncoder()
        data_file[target_col] = le.fit_transform(data_file[target_col])
        data = data_file.groupby(sample_col)
        target = [sample[1].values for sample in data[target_col]]
        features = [sample[1].drop([sample_col, target_col], axis=1, inplace=False).values for sample in data]
        return Bunch(data=features, target=target,
                     target_names=le.classes_), le


    plankton, le = load_plankton_file('/Users/albertocastano/Dropbox/PlataformaCuantificación/plancton.csv')
    # executed on client only; variables created below, including modules imported,
    # are not available in job computations

    import pickle
    with open('plankton.pkl', 'wb') as fh:
        pickle.dump(plankton, fh)

    import dispy, random

    # distribute 'compute' to nodes; 'compute' does not have any dependencies (needed from client)
    cluster = dispy.SharedJobCluster(compute, scheduler_node="dhcp015.aic.uniovi.es", depends=['plankton.pkl'])
    # run 'compute' with 20 random numbers on available CPUs
    jobs = []
    for i in range(20):
        job = cluster.submit(random.randint(10, 20), 'plankton.pkl')
        job.id = i  # associate an ID to identify jobs (if needed later)
        jobs.append(job)
    # cluster.wait() # waits until all jobs finish
    for job in jobs:
        host, n, bla = job()  # waits for job to finish and returns results
        print('%s executed job %s at %s with %s and %s' % (host, job.id, job.start_time, n, bla))
        # other fields of 'job' that may be useful:
        # job.stdout, job.stderr, job.exception, job.ip_addr, job.end_time
    cluster.print_status()  # shows which nodes executed how many jobs etc.
