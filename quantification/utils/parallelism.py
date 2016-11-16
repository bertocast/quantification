import logging

import dispy
import numpy as np

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

        cluster = dispy.SharedJobCluster(self.compute, depends=self.dependencies, loglevel=logging.ERROR,
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
        cluster.wait()

        for job in jobs:
            job()
            if job.exception is not None:
                raise ClusterException(job.exception + job.ip_addr)
            results.append(job.result)
        if self.verbose:
            cluster.print_status()
        cluster.close()
        return np.array(results)




