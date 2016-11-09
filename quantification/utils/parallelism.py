import logging

import dispy
import numpy as np

from quantification.utils.errors import ClusterException


class ClusterParallel:
    def __init__(self, compute, iter_params, constant_named_params, dependencies=None, local=False, verbose=False):
        self.compute = compute
        self.iter_params = iter_params
        self.constant_params = constant_named_params
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
            return results

        cluster = dispy.JobCluster(self.compute, depends=self.dependencies, loglevel=logging.ERROR,
                                   pulse_interval=10, reentrant=True)

        jobs = []
        for sample in self.iter_params:
            if isinstance(sample, tuple):
                job = cluster.submit(*sample, **self.constant_params)
            else:
                job = cluster.submit(sample, **self.constant_params)
            jobs.append(job)
        cluster.wait()

        for job in jobs:
            job()
            if job.exception is not None:
                raise ClusterException(job.exception)
            results.append(job.result)
        if self.verbose:
            cluster.print_status()
        cluster.close()
        return np.array(results)




