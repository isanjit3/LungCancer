
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
import glob
import os
import config

study_folders = glob.glob(os.path.join(config.DATA_PATH, "*", "*"))
print(study_folders)
