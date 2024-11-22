from multiprocessing import Manager, Queue, Lock, Barrier, Value,log_to_stderr
import ctypes
from GPUHandler import *
import logging
import numpy as np
import json
from collections import defaultdict
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def convert_numpy_int64(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()    
    elif isinstance(obj, np.int32):
        return int(obj)
    else:
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


class ConcurrencyManager:
    """
        @brief: Class to manage the concurrency of the processes
        @Description:
            each of the variables of the class have an update method that is going to be used to 
            thread safe update the variables.
            NOTE: A message for debugging purposes can be passed to the update method.
            example:
                UpdateSharedDict(key, value, Message = "Updating the shared dictionary")
                This is going to automatically call:
                    - AcquireLocksByName("SharedDictLock",Message)
                    - ReleaseLockByName("SharedDictLock",Message)
            NOTE: The same applies for the rest of the variables and the front-end just call whatever with the message
            This makes the code clearer in the multiple_launches section.
    """
    def __init__(self,num_processes,save_dir,debug = True):
        # Where save infos
        self.LogStderr = log_to_stderr()
        self.save_dir = save_dir
        self.Debug = debug
        self.manager = Manager()
        self.shared_dict = self.manager.dict()
        self.queue = Queue()
        self.barrier = Barrier(num_processes)
        self.num_processes = num_processes
        self.GPUHandler = GPUHandler()
        # Initialize a lock for each GPU        
        self.locks = [Lock() for i in range(self.GPUHandler.deviceCount)]
        self.GPUErrorLocks = [Lock() for i in range(self.GPUHandler.deviceCount)]
        self.GPULock = Lock()
        self.lock_Config_ini = Lock()   
        self.LockSharedDict = Lock()
        self.LockLogger = Lock()
        self.LockDockerError = Lock()
        # Initialize a flag for each GPU
        self.GPU_errors = [Value('b', False) for i in range(self.GPUHandler.deviceCount)]
        self.Docker_error = Value('b', False)
        self.R_errors = [Value('i', 0) for i in range(self.GPUHandler.deviceCount)]
        self.GPU_error_index = Value('i', -1) 
        pass

    #### SAVE SHARED DICTIONARY ####
    def save_shared_dict(self):
        with open(os.path.join(self.save_dir, 'shared_dict.json'), 'w') as f:
            json.dump(dict(self.shared_dict), f,default=convert_numpy_int64,indent=4)


    

    ###### ACQUIRE AND RELEASE LOCKS #####
    def AcquireLockByName(self, LockName, gpu_id = None,Message = None):
        """
            @brief: Acquire the locks for the GPUs
        """
        if LockName == "GPULock":
            self.AcquireGPULock(Message)
        elif LockName == "ConfigIniLock":
            self.AcquireConfigIniLock(Message)
        elif LockName == "LoggerLock":
            self.AcquireLoggerLock(Message)
        elif LockName == "SharedDictLock":
            self.LockSharedDict.acquire()
        elif LockName == "DockerErrorLock":
            self.AcquireDockerErrorLock(Message)
        elif LockName == "ErrorLock":
            assert gpu_id is not None, "ErrorLock: GPU ID is not provided"
            self.AcquireErrorLock(gpu_id,Message)
        elif LockName == "Locks":
            assert gpu_id is not None, "AcquireLocks: GPU ID is not provided"
            self.AcquireLocks(gpu_id,Message)
        else:
            raise ValueError("LockName not found")
        pass


    def ReleaseLockByName(self, LockName, gpu_id = None,Message = None):
        """
            @brief: Release the locks for the GPUs
        """
        if LockName == "GPULock":
            self.ReleaseGPULock(Message)
        elif LockName == "ConfigIniLock":
            self.ReleaseConfigIniLock(Message)
        elif LockName == "LoggerLock":
            self.ReleaseLoggerLock(Message)
        elif LockName == "SharedDictLock":
            self.LockSharedDict.release()
        elif LockName == "DockerErrorLock":
            self.ReleaseDockerErrorLock(Message)
        elif LockName == "ErrorLock":
            assert gpu_id is not None, "ErrorLock: GPU ID is not provided"
            self.ReleaseErrorLock(gpu_id,Message)
        elif LockName == "Locks":
            assert gpu_id is not None, "ReleaseLocks: GPU ID is not provided"
            self.ReleaseLocks(gpu_id,Message)
        else:
            raise ValueError("LockName not found")

    def AcquireLocks(self, gpu_id,Message = None):
        """
            @brief: Acquire the locks for the GPUs
        """
        self.locks[gpu_id].acquire()
        if self.Debug:
            if Message is not None:
                with self.LockLogger:
                    logger.info(Message)
        pass

    def ReleaseLocks(self, gpu_id,Message = None):
        """
            @brief: Release the locks for the GPUs
        """
        self.locks[gpu_id].release()
        if self.Debug:
            if Message is not None:
                with self.LockLogger:
                    logger.info(Message)
                
        pass

    def AcquireGPULock(self,Message = None):
        """
            @brief: Acquire the lock for the GPUs
        """
        self.GPULock.acquire()
        if self.Debug:
            if Message is not None:
                with self.LockLogger:
                    logger.info(Message)

        pass

    def ReleaseGPULock(self,Message = None):
        """
            @brief: Release the lock for the GPUs
        """
        self.GPULock.release()
        if self.Debug:
            if Message is not None:
                with self.LockLogger:
                    logger.info(Message)
        pass

    def AcquireConfigIniLock(self,Message = None):
        """
            @brief: Acquire the lock for the Config.ini
        """
        self.lock_Config_ini.acquire()
        if self.Debug:
            if Message is not None:
                with self.LockLogger:
                    logger.info(Message)

        pass

    def ReleaseConfigIniLock(self,Message = None):
        """
            @brief: Release the lock for the Config.ini
        """
        self.lock_Config_ini.release()
        if self.Debug:
            if Message is not None:
                with self.LockLogger:
                    logger.info(Message)

        pass

    def AcquireLoggerLock(self,Message = None):
        """
            @brief: Acquire the lock for the Logger
        """
        self.LockLogger.acquire()
        if self.Debug:
            if Message is not None:
                with self.LockLogger:
                    logger.info(Message)

        pass

    def ReleaseLoggerLock(self,Message = None):
        """
            @brief: Release the lock for the Logger
        """
        self.LockLogger.release()
        if self.Debug:
            if Message is not None:
                with self.LockLogger:
                    logger.info(Message)

        pass


    def AcquireErrorLock(self, gpu_id,Message = None):
        """
            @brief: Acquire the lock for the Errors
        """
        self.GPUErrorLocks[gpu_id].acquire()
        if self.Debug:
            if Message is not None:
                with self.LockLogger:
                    logger.info(Message)

        pass

    def ReleaseErrorLock(self, gpu_id,Message = None):
        """
            @brief: Release the lock for the Errors
        """
        self.GPUErrorLocks[gpu_id].release()
        if self.Debug:
            if Message is not None:
                with self.LockLogger:
                    logger.info(Message)

        pass

    def AcquireDockerErrorLock(self,Message = None):
        """
            @brief: Acquire the lock for the Docker
        """
        self.LockDockerError.acquire()
        if self.Debug:
            if Message is not None:
                with self.LockLogger:
                    logger.info(Message)

        pass
    
    def ReleaseDockerErrorLock(self,Message = None):
        """
            @brief: Release the lock for the Docker
        """
        self.LockDockerError.release()
        if self.Debug:
            if Message is not None:
                with self.LockLogger:
                    logger.info(Message)

        pass
    ##### UPDATE SHARED VARIABLES #####
    def InitSharedDict(self,pid,case,R,UCI1,city,Message = None):
        self.AcquireLockByName("SharedDictLock",Message = Message)
        self.shared_dict = defaultdict(dict)
        self.shared_dict[pid] = {
            'name': case,
            'R': R,
            'UCI': round(UCI1,3),
            'city': city,
            'GPU_occupied': False,
            "GPU_error": False,
            "docker_error": False,
            "failed_simulation": False
        }
        self.save_shared_dict()
        self.ReleaseLockByName("SharedDictLock",Message = Message)

    def UpdateSharedDict(self, pid,key, value,Message = None):
        """
            @brief: Update the shared dictionary
        """
        self.AcquireLockByName("SharedDictLock",Message = Message)
        self.shared_dict[pid] = self.shared_dict.get(pid, {})
        self.shared_dict[pid][key] = value
        self.save_shared_dict()
        self.ReleaseLockByName("SharedDictLock",Message = Message)

    def UpdateDockerError(self, error,Message = None):
        """
            @brief: Update the error for the Docker
        """
        self.AcquireLockByName("DockerErrorLock",Message = Message)
        self.Docker_error.value = error
        self.ReleaseLockByName("DockerErrorLock",Message = Message)
        pass


    def UpdateGPUError(self, gpu_id, error,Message = None):
        """
            @brief: Update the error for the GPUs
        """
        self.AcquireLockByName("ErrorLock",gpu_id,Message = Message)
        self.GPU_errors[gpu_id].value = error
        self.ReleaseLockByName("ErrorLock",gpu_id,Message = Message)
        pass


    def UpdateRError(self, gpu_id, R,Message = None):
        """
            @brief: Update the error for the R
        """
        self.AcquireLockByName("ErrorLock",gpu_id,Message = Message)
        self.R_errors[gpu_id].value = R
        self.ReleaseLockByName("ErrorLock",gpu_id,Message = Message)        
        pass        
        
    def UpdateGPUErrorIndex(self, gpu_id,Message = None):
        """
            @brief: Update the index of the GPU error
        """
        self.AcquireLockByName("ErrorLock",gpu_id,Message = Message)
        self.GPU_error_index.value = gpu_id
        self.ReleaseLockByName("ErrorLock",gpu_id,Message = Message)
        pass 

    #### LOGGING ####
    def LogMessage(self, messages,Message = None):
        """
            @brief: Log the message
        """
        self.AcquireLockByName("LoggerLock",Message = Message)
        for message in messages:
            logger.info(message)
        self.ReleaseLockByName("LoggerLock",Message = Message)
        pass

##### RESET `SHARED VARIABLES #####

    def Reset(self):
        """
            @brief: Reset the shared dictionary
        """
#        self.GPUHandler.Refresh()
        self.locks = [Lock() for i in range(self.GPUHandler.deviceCount)]   
        # Initialize a flag for each GPU
        self.GPU_errors = [Value('b', False) for i in range(self.GPUHandler.deviceCount)]
        self.Docker_error = Value('b', False)
        self.R_errors = [Value('i', 0) for i in range(self.GPUHandler.deviceCount)]
        self.GPU_error_index = Value('i', -1) 

        pass
    