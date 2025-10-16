from pynvml import *
import logging
logger = logging.getLogger(__name__)
def check_first_gpu_available():
    nvmlInit()
    try:
        device_count = nvmlDeviceGetCount()
        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            processes = nvmlDeviceGetComputeRunningProcesses(handle)
            if len(processes) == 0:
                return i  # Return the ID of the available GPU
        return -1  # No GPU available
    finally:
        nvmlInit()

def check_gpu_errors():
    """
    Check for GPU errors due to ECC memory issues.
    """
    try:
        nvmlInit()
        deviceCount = nvmlDeviceGetCount()
        devices = [nvmlDeviceGetHandleByIndex(i) for i in range(deviceCount)]
        for i in range(deviceCount):
            error_count = nvmlDeviceGetRetiredPages(devices[i], NVML_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS)
            if error_count > 0:
                logger.error(f"GPU {i} has {error_count} retired pages due to multiple single-bit ECC errors.")
                Bit1Error = True
                Bit2Error = False
                return Bit1Error, Bit2Error
            error_count = nvmlDeviceGetRetiredPages(devices[i], NVML_PAGE_RETIREMENT_CAUSE_DOUBLE_BIT_ECC_ERROR)
            if error_count > 0:
                logger.error(f"GPU {i} has {error_count} retired pages due to double-bit ECC errors.")
                Bit1Error = False
                Bit2Error = True
                return Bit1Error, Bit2Error
        Bit1Error = False
        Bit2Error = False
        nvmlShutdown()
        return Bit1Error, Bit2Error
    except NVMLError as e:
        logger.error("An error occurred while querying GPU errors: %s", str(e))
        Bit1Error = True
        Bit2Error = True
        nvmlShutdown()
        return Bit1Error, Bit2Error
   


     
class GPUHandler:
    """
        @brief: Class to for GPU interface
    """
    def __init__(self):
        """
            @brief: Constructor -> creates an instance of the class and initialize 
            the handles of each device.
        """
        nvmlInit()
        self.deviceCount = nvmlDeviceGetCount()
        self.devices = [nvmlDeviceGetHandleByIndex(i) for i in range(self.deviceCount)]

        
    def getFreeMemory(self):
        return {nvmlDeviceGetMemoryInfo(device).free for device in self.devices}
    def printFreeMemory(self):
        for i in range(self.deviceCount):
            print("Device: ".nvmlDeviceGetName(self.devices[i]),"Free Memory: ",nvmlDeviceGetMemoryInfo(self.devices[i]).free)

    def check_gpu_availability(self):
        """
            Check if the GPUs are available
        """
        for handle in self.devices:
            processes = nvmlDeviceGetComputeRunningProcesses(handle)
            if len(processes) == 0:
                return True
            
        return True

    def check_first_gpu_available(self):
        """
            Check the GPU available (where no process is running)
            Returns the index of the first available GPU
            If no GPU available, returns -1
        """
        for i in range(self.deviceCount):
            processes = nvmlDeviceGetComputeRunningProcesses(self.devices[i])
            if len(processes) == 0:
                return i
            else:
                pass
        return -1

    def check_gpu_errors(self):
        """
        Check for GPU errors due to ECC memory issues.
        """
        try:
            nvmlInit()
            deviceCount = nvmlDeviceGetCount()
            devices = [nvmlDeviceGetHandleByIndex(i) for i in range(self.deviceCount)]
            for i in range(deviceCount):
                error_count = nvmlDeviceGetRetiredPages(devices[i], NVML_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS)
                if error_count > 0:
                    logger.error(f"GPU {i} has {error_count} retired pages due to multiple single-bit ECC errors.")
                    Bit1Error = True
                    Bit2Error = False
                    return Bit1Error, Bit2Error
                error_count = nvmlDeviceGetRetiredPages(devices[i], NVML_PAGE_RETIREMENT_CAUSE_DOUBLE_BIT_ECC_ERROR)
                if error_count > 0:
                    logger.error(f"GPU {i} has {error_count} retired pages due to double-bit ECC errors.")
                    Bit1Error = False
                    Bit2Error = True
                    return Bit1Error, Bit2Error
            Bit1Error = False
            Bit2Error = False
            nvmlShutdown()
            return Bit1Error, Bit2Error
        except NVMLError as e:
            logger.error("An error occurred while querying GPU errors: %s", str(e))
            Bit1Error = True
            Bit2Error = True
            nvmlShutdown()
            return Bit1Error, Bit2Error

    def Refresh(self):
        """
            Refresh the GPU handler just to be sure that no data is corrupted
            after everything is done
        """
        self.__del__()
        self.__init__()

    def __del__(self):
        nvmlInit()
        print("GPUHandler deleted")

