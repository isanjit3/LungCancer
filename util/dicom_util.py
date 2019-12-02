#DICOM utility class
import os
import pydicom
import numpy as np
import dicom_contour.contour as dcm
import matplotlib.pyplot as plt


#This method returns the image in the HU Units
def parse_dicom_file(fileName):
  """Parse the given DICOM filename
    :param filename: filepath to the DICOM file to parse
    :return: dictionary with DICOM image data
    """
  try:
    dcm = pydicom.read_file(fileName)
    dcm_image = dcm.pixel_array

    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    dcm_image = dcm_image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    dcm_image[dcm_image == -2000] = 0

    try:
        intercept = dcm.RescaleIntercept        
    except AttributeError:        
        intercept = 0.0

    try:
        slope = dcm.RescaleSlope        
    except AttributeError:        
        slope = 0.0
    
    if slope != 1:
      dcm_image = slope * dcm_image.astype(np.float64)
      dcm_image = dcm_image.astype(np.int16)            
      dcm_image += np.int16(intercept)

    return np.array(dcm_image, dtype=np.int16)

  except:
    return None






def get_roi_contour_ds(rt_sequence, index):
    """
    Extract desired Regions of Interest(ROI) contour datasets
    from RT Sequence.
    
    E.g. rt_sequence can have contours for different parts of the lung
    
    You can use get_roi_names to find which index to use
    
    Inputs:
        rt_sequence (pydicom.dataset.FileDataset): Contour file dataset, what you get 
                                                after reading contour DICOM file
        index (int): Index for ROI Sequence
    Return:
        contours (list): list of ROI contour pydicom.dataset.Dataset s
    """
    if (len(rt_sequence.ROIContourSequence) == 0):
      return []

    # index 0 means that we are getting RTV information
    ROI = rt_sequence.ROIContourSequence[index]
    # get contour datasets in a list
    contours = [contour for contour in ROI.ContourSequence]
    return contours





def contour2poly(contour_dataset, path,slices_imgpath_dict):
    """
    Given a contour dataset (a DICOM class) and path that has .dcm files of
    corresponding images return polygon coordinates for the contours.

    Inputs
        contour_dataset (pydicom.dataset.Dataset) : DICOM dataset class that is identified as
                        (3006, 0016)  Contour Image Sequence
        path (str): path of directory containing DICOM images

    Return:
        pixel_coords (list): list of tuples having pixel coordinates
        img_ID (id): DICOM image id which maps input contour dataset
        img_shape (tuple): DICOM image shape - height, width
    """

    contour_coord = contour_dataset.ContourData
    # x, y, z coordinates of the contour in mm
    
    coord = []
    for i in range(0, len(contour_coord), 3):
        coord.append((contour_coord[i], contour_coord[i + 1], contour_coord[i + 2]))

    # extract the image id corresponding to given countour
    # read that dicom file
    img_ID = contour_dataset.ContourImageSequence[0].ReferencedSOPInstanceUID
    
    if (img_ID not in slices_imgpath_dict):
      print("Image ID:", img_ID, "not found in the slice image path dict.")
      return;
    
    img = pydicom.read_file(os.path.join(path, slices_imgpath_dict[img_ID]))
    img_arr = img.pixel_array
    img_shape = img_arr.shape
    
    # physical distance between the center of each pixel
    x_spacing, y_spacing = float(img.PixelSpacing[0]), float(img.PixelSpacing[1])

    # this is the center of the upper left voxel
    origin_x, origin_y, _ = img.ImagePositionPatient

    # y, x is how it's mapped
    pixel_coords = [(np.ceil((x - origin_x) / x_spacing), np.ceil((y - origin_y) / y_spacing))  for x, y, _ in coord]
    return pixel_coords, img_ID, img_shape





def poly_to_mask(polygon, width, height):
    from PIL import Image, ImageDraw
    
    """Convert polygon to mask
    :param polygon: list of pairs of x, y coords [(x1, y1), (x2, y2), ...]
    in units of pixels
    :param width: scalar image width
    :param height: scalar image height
    :return: Boolean mask of shape (height, width)
    """

    # http://stackoverflow.com/a/3732128/1410871
    img = Image.new(mode='L', size=(width, height), color=0)
    ImageDraw.Draw(img).polygon(xy=polygon, outline=0, fill=1)
    mask = np.array(img).astype(bool)
    return mask





def get_mask_dict(contour_datasets, path, slices_imgpath_dict):
    """
    Inputs:
        contour_datasets (list): list of pydicom.dataset.Dataset for contours
        path (str): path of directory with images

    Return:
        img_contours_dict (dict): img_id : contour array pairs
    """
    
    from collections import defaultdict
    
    # create empty dict for 
    img_contours_dict = defaultdict(int)

    for cdataset in contour_datasets:
        coords, img_id, shape = contour2poly(cdataset, path, slices_imgpath_dict) or (None, None, None)
        if coords is None:
          continue
          
        mask = poly_to_mask(coords, *shape)
        img_contours_dict[img_id] += mask
    
    return img_contours_dict





def get_img_mask_voxel(slice_orders, mask_dict, image_path, slices_imgpath_dict):
    """ 
    Construct image and mask voxels
    
    Inputs:
        slice_orders (list): list of tuples of ordered img_id and z-coordinate position
        mask_dict (dict): dictionary having img_id : contour array pairs
        image_path (str): directory path containing DICOM image files
    Return: 
        img_voxel: ordered image voxel for CT/MR
        mask_voxel: ordered mask voxel for CT/MR
    """
    
    img_voxel = []
    mask_voxel = []
    tumor_only_slices = []
    for img_id, _ in slice_orders:
        path = os.path.join(image_path, slices_imgpath_dict[img_id])
        img_array = parse_dicom_file(path)
        
        if img_id in mask_dict: 
          mask_array = mask_dict[img_id]
          if (np.count_nonzero(mask_array) > 0):
            tumor_only_slices.append(1)
          else:
            tumor_only_slices.append(0)
        else: 
          mask_array = np.zeros_like(img_array)
          tumor_only_slices.append(0)
          
        img_voxel.append(img_array)
        mask_voxel.append(mask_array)
        
    return img_voxel, mask_voxel, tumor_only_slices





def show_img_msk_fromarrayX(img_arr, msk_arr, alpha=0.35, sz=7, cmap='inferno',
                           save_path=None):

    """
    Show original image and masked on top of image
    next to each other in desired size
    Inputs:
        img_arr (np.array): array of the image
        msk_arr (np.array): array of the mask
        alpha (float): a number between 0 and 1 for mask transparency
        sz (int): figure size for display
        save_path (str): path to save the figure
    """

    msk_arr = np.ma.masked_where(msk_arr == 0, msk_arr)
    plt.figure(figsize=(sz, sz))
    plt.subplot(1, 2, 1)
    plt.imshow(img_arr, cmap='gray')
    plt.imshow(msk_arr, cmap=cmap, alpha=alpha)
    plt.subplot(1, 2, 2)
    plt.imshow(img_arr, cmap='gray')
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()





def slice_order(path):
    """
    Takes path of directory that has the DICOM images and returns
    a ordered list that has ordered filenames
    Inputs
        path: path that has .dcm images
    Returns
        ordered_slices: ordered tuples of filename and z-position
    """
    # handle `/` missing
    if path[-1] != '/': path += '/'
    slices = []
    slices_img_path = {}
    for s in os.listdir(path):
        try:
            f = pydicom.read_file(path + '/' + s)
            f.pixel_array  # to ensure not to read contour file
            slices.append(f)
            slices_img_path[f.SOPInstanceUID] = s
        except:
            continue

    slice_dict = {s.SOPInstanceUID: s.ImagePositionPatient[-1] for s in slices}
    ordered_slices = sorted(slice_dict.items(), key=dcm.operator.itemgetter(1))
    return ordered_slices,slices_img_path





def show_img(im, figsize=None, ax=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax





def get_scan_mask_data(image_path, contour_filename, return_tumor_only_slices = True):
  # read dataset for contour
  rt_sequence = pydicom.read_file(contour_filename)

  # get contour datasets with index idx
  idx = 0
  contour_datasets = get_roi_contour_ds(rt_sequence, idx)

  # get slice orders
  slice_orders, slices_imgpath_dict = slice_order(image_path)

 
  # construct mask dictionary
  mask_dict = get_mask_dict(contour_datasets, image_path, slices_imgpath_dict)

  # get image and mask data for patient
  img_data, mask_data, tumor_only_slices = get_img_mask_voxel(slice_orders, mask_dict, image_path, slices_imgpath_dict)
  if return_tumor_only_slices and (1 in tumor_only_slices):  
    idx = tumor_only_slices.index(1)  
    ldx = max(idx for idx, val in enumerate(tumor_only_slices) if val == 1) 
    return img_data[idx:ldx], mask_data[idx:ldx] # Return only the tumor slices
  else: 
    return img_data, mask_data
  





def get_roi_names(contour_data):
    """
    This function will return the names of different contour data,
    e.g. different contours from different experts and returns the name of each.
    Inputs:
        contour_data (dicom.dataset.FileDataset): contour dataset, read by dicom.read_file
    Returns:
        roi_seq_names (list): names of the
    """
    roi_seq_names = [roi_seq.ROIName for roi_seq in list(contour_data.StructureSetROISequence)]
    return roi_seq_names





def show_img_msk_fromarray(img_arr, msk_arr, alpha=0.35, sz=7, cmap='inferno',
                           save_path=None):

    """
    Show original image and masked on top of image
    next to each other in desired size
    Inputs:
        img_arr (np.array): array of the image
        msk_arr (np.array): array of the mask
        alpha (float): a number between 0 and 1 for mask transparency
        sz (int): figure size for display
        save_path (str): path to save the figure
    """

    #msk_arr = np.ma.masked_where(msk_arr == 0, msk_arr)
    plt.figure(figsize=(sz, sz))
    plt.subplot(1, 2, 1)
    plt.imshow(img_arr, cmap='gray')
    #plt.imshow(msk_arr, cmap=cmap, alpha=alpha)
    #plt.imshow(msk_arr, cmap='jet', interpolation='none', alpha=None)
    plt.subplot(1, 2, 2)
    #plt.imshow(img_arr, cmap='gray')
    plt.imshow(msk_arr, cmap='jet', interpolation='none', alpha=None)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()





def resample_img(image, scan, new_spacing=[1,1,1]):
  """
A scan may have a pixel spacing of [2.5, 0.5, 0.5], which means that the distance 
between slices is 2.5 millimeters. For a different scan this may be [1.5, 0.725, 0.725], 
this can be problematic for automatic analysis (e.g. using ConvNets)!

A common method of dealing with this is resampling the full dataset to a certain 
isotropic resolution. If we choose to resample everything to 1mm1mm1mm pixels we 
can use 3D convnets without worrying about learning zoom/slice thickness invariance.

Whilst this may seem like a very simple step, it has quite some edge cases due to 
rounding. Also, it takes quite a while.
"""
  # Determine current pixel spacing
  spacing = np.array([scan[0].SliceThickness] + list(scan[0].PixelSpacing), dtype=np.float32)

  resize_factor = spacing / new_spacing
  new_real_shape = image.shape * resize_factor
  new_shape = np.round(new_real_shape)
  real_resize_factor = new_shape / image.shape
  new_spacing = spacing / real_resize_factor
  
  image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

  return image, new_spacing






def normalize(image, min_bound, max_bound, pixel_mean):
    image = (image - min_bound) / (max_bound - min_bound)
    image[image>1] = 1.
    image[image<0] = 0.

    #zero center the image so that the mean value is 0
    image = image - pixel_mean
    return image





def get_scan_slices(path):
  slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
  slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
  try:
      slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
  except:
      slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
  for s in slices:
      s.SliceThickness = slice_thickness
        
  return slices