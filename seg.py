import numpy as np
import glob
import os
import tensorflow as tf
import keras
from util import dicom_util
from util import data_util
import config
from model import unet

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.7
tf.keras.backend.set_session(tf.Session(config=tf_config))



"""
slice_orders, slices_imgpath_dict = dicom_util.slice_order(config.image_path)

print(slice_orders[0])
img_data, mask_data = dicom_util.get_scan_macd sk_data(config.image_path, config.contour_filename)
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

model = unet.build_cnn_unet_model()

# Fit model
if os.path.exists('model-lung-segmentation.h5'):
  print("Model Already exists. Skipping the training.")
else:
  earlystopper = keras.callbacks.EarlyStopping(patience=5, verbose=1)
  checkpointer = keras.callbacks.ModelCheckpoint('model-lung-segmentation.h5', verbose=1, save_best_only=True)
  results = model.fit(X_train, Y_train, validation_split=0.2, batch_size=4, epochs=10, 
                    callbacks=[earlystopper, checkpointer], shuffle=True)
                    