import pickle
import mmap
import os
import struct
import numpy as np
import time
import random
import signal

import sys
import multiprocessing
from multiprocessing import Process
from multiprocessing import current_process

# this python script is responsible to host the connections from other applications, in a shared memory manner


# MACOS run with OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES env before
# example: OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES python3 shared_memory_comm.py

def create_file(filename,size):
    with open(filename, 'wb') as f:
        f.seek(size-1)
        f.write(b'\x00')
        f.close()

def memory_map(filename, access=mmap.ACCESS_WRITE):
    size = os.path.getsize(filename)
    fd = os.open(filename, os.O_RDWR)
    return mmap.mmap(fd, size, access=access)


class SharedMemoryInterfaceProcess(multiprocessing.Process):

    def __init__(self, num, conf_dict):
        multiprocessing.Process.__init__(self)
        self.exit = multiprocessing.Event()
        self.num = num
        self.ndim_x = conf_dict['ndim_x']
        self.is_emm = conf_dict['is_emm']
        self.model_address = conf_dict['model_address']
        self.mm_filename = 'sm_dvp_' + str(num) + '.dat'

        create_file(self.mm_filename,40) # 5 doubles are stored
        self.mmap_obj = memory_map(self.mm_filename)

    def run(self):

        from cde.density_estimator import NoNaNGPDExtremeValueMixtureDensityNetwork
        from cde.density_estimator import MixtureDensityNetwork
        
        if self.is_emm:
            self.model = NoNaNGPDExtremeValueMixtureDensityNetwork(name='EMM_'+str(self.num), ndim_x=self.ndim_x, ndim_y=1)
        else:
            self.model = MixtureDensityNetwork(name="GMM-"+str(self.num), ndim_x=self.ndim_x, ndim_y=1)

        self.model._setup_inference_and_initialize()
        
        with open(self.model_address, 'rb') as input:
            self.model = pickle.load(input)
        
        print('Estimator '+str(self.num)+' is up and ready.')


        while not self.exit.is_set():

            # wait for MATLAB to write
            while not self.exit.is_set():
                self.mmap_obj.seek(0)
                res = struct.unpack('ddddd', self.mmap_obj.read())
                if res[0] == 1.000:
                    break
                else:
                    time.sleep(0.0001)

            #prepare mx and my
            mx = [res[1]]
            my = [res[4]]

            # get the result
            mx2 = np.array([mx])
            my2 = np.array([my])
            res = self.model.tail(mx2,my2)
            ans = res.item()

            self.mmap_obj.seek(0)
            self.mmap_obj.write(struct.pack('ddddd', 0.0, ans, 0.0, 0.0, 0.0))

        print('Memory-mapped TF-MATLAB interface process ' + str(self.num) + ' exited.')

    def shutdown(self):
        self.exit.set()

def signal_handler(sig, frame):
    if (current_process().name == 'MainProcess'):
        print('You pressed Ctrl+C')
        for process in processes:
            process.shutdown()

        for process in processes:
            print("Child process state: %d" % process.is_alive())

        for process in processes:
            process.join()

        sys.exit()

if __name__ == "__main__":

    # define processes as global
    # you can define as many as CPU cores of the system
    conf = { 1 : {
            "is_emm" : True,
            "ndim_x" : 3,
            "model_address" : "trained_emm_p3_s25_r0.pkl",
        }, 2 : { 
            "is_emm" : True,
            "ndim_x" : 3,
            "model_address" : "trained_emm_p3_s25_r0.pkl",
        } 
    }

    processes = [SharedMemoryInterfaceProcess(num, conf[num]) for num in conf.keys()]

    signal.signal(signal.SIGINT, signal_handler)
    print('Press Ctrl+C for terminating the processes')

    for process in processes:
        process.start()

    #time.sleep(120)

    #for process in processes:
    #    process.shutdown()

    #time.sleep(3)
    #for process in processes:
    #    print("Child process state: %d" % process.is_alive())

    #for process in processes:
    #    process.join()


    
