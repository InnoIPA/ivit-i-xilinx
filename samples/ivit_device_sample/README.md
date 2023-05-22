# IVIT-I iDEVICE 
iVIT-I iDEVICE Sample, this sample demonstrates how to do use iDEVICE to monitor device on [iVIT](../../README.md).

# Usage
* Create instance for iDevice.
    ```bash
    #import iDevice from ivit
    from ivit_i.utils import iDevice 

    idev = iDevice()

    ```
* Another useful function.  
    1. Use `idev.get_device_info()` can get all device information, and the format of return like below.  

    

        ```bash

        {   #CPU information.
            'DPU':{
                'id': 0,                # the idex wget from device. 
                'uuid': 'DPU',          # the name get from device. 
                'load': 0,              # loading capacity get from device. 
                'memoryUtil': 0,        # amount of memory usage get from device. 
                'temperature': 30       # temperature get from device
            }
        }
        ```
    2. Use `idev.get_device_info('DPU')` can get target device information,and the format of return like below.

        ```json
        {
            "DPU": {
                    "id": 0, 
                    "uid": "DPU", 
                    "load": 0, 
                    "memoryUtil": 0, 
                    "temperature": 40.33333
            }
        }
        ```
    