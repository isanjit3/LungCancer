import glob
import os
import pydicom
import numpy as np
from util import dicom_util
import config

def get_study_images(study_folders, load_tumor_only_slices = True):  
  x=[]
  y=[]
  x_temp = []
  y_temp = []
  for si, sf in enumerate(study_folders):
    scan_folders = glob.glob(os.path.join(sf, "*"))
    
    if (len(scan_folders) == 2):
      scan_image_path = ''
      contour_file_name = ''
      scan = os.listdir(scan_folders[0])[0]
      file_name = os.path.join(scan_folders[0], scan)
    
      try:
        dicom_file = pydicom.read_file(file_name)     
        dicom_file.pixel_array #Check if this is scan file.
        scan_image_path = scan_folders[0]
        contour_file_name = os.path.join(scan_folders[1], os.listdir(scan_folders[1])[0])      
      except:
        #This is contour file.     
        scan_image_path = scan_folders[1]     
        contour_file_name = file_name     
      
      print('Processing Scan path: ', scan_image_path)       
      img_data, mask_data = dicom_util.get_scan_mask_data(scan_image_path, contour_file_name, load_tumor_only_slices)
      print("Scan Image length:", len(img_data))
      #Convert to numpy array for easy processing.
      np_img = np.asarray(img_data, dtype=np.float16)
      np_mask = np.asarray(mask_data, dtype=np.int8)

      np_img = dicom_util.normalize(np_img, config.MIN_BOUND, config.MAX_BOUND, config.PIXEL_MEAN)
      #Plot the tumor slices.
      #for img, mask in zip(img_data, mask_data):
      #  show_img_msk_fromarray(img, mask, sz=10, cmap='inferno', alpha=0.5)

      for i in range(0, len(np_img)):
        x_temp.append(np_img[i])
        y_temp.append(np_mask[i])
        
        if (len(x_temp) == 8):
          x_temp = np.array(x_temp)
          y_temp = np.array(y_temp)
          x.append(np.dstack(x_temp[0:8]))
          y.append(np.dstack(y_temp[0:8]))
          x_temp = []
          y_temp = []
      
      #TODO: Create 3D array of 512x512x8
      #Number of whole batches
      """
      batch_count = len(np_img)//8
      print("Batch: ", batch_count)
      for i in range(0, batch_count):
        x.append(np.dstack(np_img[i*8 : (i+1)*8]))
        y.append(np.dstack(np_mask[i*8 : (i+1)*8]))        
      """
  return np.array(x), np.array(y)