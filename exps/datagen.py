"""
    Batch-generate data
"""

import os
import sys
import numpy as np
import multiprocessing as mp
from subprocess import call
import time
from utils import printout

class DataGen(object):

    def __init__(self, env_name, num_processes, flog=None):
        self.env_name = env_name
        self.num_processes = num_processes
        self.flog = flog
        
        self.todos = []
        self.processes = []
        self.is_running = False
        self.Q = mp.Queue()

    def __len__(self):
        return len(self.todos)

    def add_one_collect_job(self, data_dir, data_split, shape_id, category, epoch_id, trial_id):
        if self.is_running:
            printout(self.flog, 'ERROR: cannot add a new job while DataGen is running!')
            exit(1)

        todo = ('COLLECT', data_dir, data_split, shape_id, category, epoch_id, trial_id, np.random.randint(10000000), self.env_name)
        self.todos.append(todo)
    
    @staticmethod
    def job_func(pid, todos, Q):
        succ_todos = []
        for todo in todos:
            if todo[0] == 'COLLECT':
                #cmd = 'xvfb-run -a python collect_data.py %s %s --out_dir %s --data_split %s --epoch_id %d --trial_id %d --random_seed %d --no_gui > /dev/null 2>&1' \
                #cmd = 'xvfb-run -a python env_%s/collect_data.py %s %s --out_dir %s --data_split %s --epoch_id %d --trial_id %d --random_seed %d --no_gui' \
                cmd = 'python env_%s/collect_data.py %s %s --out_dir %s --data_split %s --epoch_id %d --trial_id %d --random_seed %d --no_gui' \
                        % (todo[-1], todo[3], todo[4], todo[1], todo[2], todo[5], todo[6], todo[7])
                folder_name = todo[1]
                job_name = '%s_%s_%d_%d' % (todo[3], todo[4], todo[5], todo[6])
            ret = call(cmd, shell=True)
            if ret == 0:
                succ_todos.append(os.path.join(folder_name, job_name))
            if ret == 2:
                succ_todos.append(None)
        Q.put(succ_todos)

    def start_all(self):
        if self.is_running:
            printout(self.flog, 'ERROR: cannot start all while DataGen is running!')
            exit(1)

        total_todos = len(self)
        num_todos_per_process = int(np.ceil(total_todos / self.num_processes))
        np.random.shuffle(self.todos)
        for i in range(self.num_processes):
            todos = self.todos[i*num_todos_per_process: min(total_todos, (i+1)*num_todos_per_process)]
            p = mp.Process(target=self.job_func, args=(i, todos, self.Q))
            p.start()
            self.processes.append(p)
        
        self.is_running = True

    def join_all(self):
        if not self.is_running:
            printout(self.flog, 'ERROR: cannot join all while DataGen is idle!')
            exit(1)

        ret = []
        for p in self.processes:
            ret += self.Q.get()

        for p in self.processes:
            p.join()

        self.todos = []
        self.processes = []
        self.Q = mp.Queue()
        self.is_running=False
        return ret


