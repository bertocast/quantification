# coding=utf-8
import logging

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

    def compute(n):
        import time, socket, pickle
        time.sleep(n)
        host = socket.gethostname()

        return (host, n)


    import dispy, random

    # distribute 'compute' to nodes; 'compute' does not have any dependencies (needed from client)
    cluster = dispy.SharedJobCluster(compute, scheduler_node="dhcp015.aic.uniovi.es")
    # run 'compute' with 20 random numbers on available CPUs
    jobs = []
    for i in range(20):
        job = cluster.submit(random.randint(10, 20))
        job.id = i  # associate an ID to identify jobs (if needed later)
        jobs.append(job)
    # cluster.wait() # waits until all jobs finish
    for job in jobs:
        host, n = job()  # waits for job to finish and returns results
        print('%s executed job %s at %s with %s and %s' % (host, job.id, job.start_time, n))
        # other fields of 'job' that may be useful:
        # job.stdout, job.stderr, job.exception, job.ip_addr, job.end_time
    cluster.print_status()  # shows which nodes executed how many jobs etc.
