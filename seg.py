import numpy as np
import glob
import os
from util import dicom_util
from util import data_util
import config

"""
slice_orders, slices_imgpath_dict = dicom_util.slice_order(config.image_path)

print(slice_orders[0])
img_data, mask_data = dicom_util.get_scan_mask_data(config.image_path, config.contour_filename)
"""

study_folders = glob.glob(os.path.join(config.DATA_PATH, "*", "*"))

#Split the studies into training dataset and test data sets. 
studies = np.array(study_folders)
np.random.shuffle(studies)
num_studies = len(studies)

#split the studies
train_study_folders, test_study_folders = studies[:(int(num_studies * 0.95))], studies[(int(num_studies * 0.95)):]
if os.path.exists('model-lung-segmentation.h5'):
  print("Model already exists. Not loading training and validation datasets.")
else:
  X_train,  Y_train = data_util.get_study_images(train_study_folders) 
  
X_test,Y_test   = data_util.get_study_images(test_study_folders, False)