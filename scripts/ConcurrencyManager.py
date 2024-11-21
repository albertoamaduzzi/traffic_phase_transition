from multiprocessing import Manager, Queue, Lock, Barrier, Value
import ctypes
from GPUHandler import *
class ConcurrencyManager:
    """
        @brief: Class to manage the concurrency of the processes

    """
    def __init__(self,num_processes,save_dir):
        # Where save infos
        self.save_dir = save_dir
        self.manager = Manager()
        self.shared_dict = self.manager.dict()
        self.queue = Queue()
        self.barrier = Barrier(num_processes)
        self.num_processes = num_processes
        self.GPUHandler = GPUHandler()
        # Initialize a lock for each GPU        
        self.locks = [Lock() for i in range(self.GPUHandler.deviceCount)]
        self.GPULock = Lock()
        self.lock_Config_ini = Lock()   
        # Initialize a flag for each GPU
        self.GPU_errors = [Value(ctypes.c_bool, False) for i in range(self.GPUHandler.deviceCount)]
        self.Docker_error = Value(ctypes.c_bool, False)
        self.R_errors = [Value(ctypes.c_int, 0) for i in range(self.GPUHandler.deviceCount)]
        self.GPU_error_index = Value(ctypes.c_int, -1) 
        pass

    def Reset(self):
        """
            @brief: Reset the shared dictionary
        """
#        self.GPUHandler.Refresh()
        self.locks = [Lock() for i in range(self.GPUHandler.deviceCount)]   
        # Initialize a flag for each GPU
        self.GPU_errors = [Value(ctypes.c_bool, False) for i in range(self.GPUHandler.deviceCount)]
        self.Docker_error = Value(ctypes.c_bool, False)
        self.R_errors = [Value(ctypes.c_int, 0) for i in range(self.GPUHandler.deviceCount)]
        self.GPU_error_index = Value(ctypes.c_int, -1) 

        pass
    